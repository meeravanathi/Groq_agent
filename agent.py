"""
LangChain Agent for E-commerce Customer Service (using Groq API)
"""
from mock_databases import MockOrderDatabase, MockProductDatabase, MockCustomerDatabase
import os
import traceback
from typing import Dict, List, Any
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from tools import get_tools  # Assumes you have a tools.py file with custom tool definitions
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.agents import AgentAction, AgentFinish
import re

class SafeOutputParser(BaseOutputParser):
    """A patched parser that avoids ReAct parser crash on malformed output"""

    def parse(self, text: str):
        try:
            if "Final Answer:" in text:
                # Return final answer
                final_answer = text.split("Final Answer:")[-1].strip()
                return AgentFinish(return_values={"output": final_answer}, log=text)
            
            # Match typical ReAct format
            action_match = re.search(r"Action: (.*)\nAction Input: (.*)", text)
            if action_match:
                return AgentAction(
                    tool=action_match.group(1).strip(),
                    tool_input=action_match.group(2).strip(),
                    log=text
                )
            
            # Default fallback
            return AgentFinish(return_values={"output": text.strip()}, log=text)
        
        except Exception as e:
            # Fallback to safe finish with raw output
            return AgentFinish(return_values={"output": f"[Unparsed Output] {text.strip()}"}, log=text)

load_dotenv()  # Load .env file if available

class EcommerceAgent:
    """E-commerce customer service agent using Groq API"""

    def __init__(self, groq_api_key: str = None):
        """Initialize the agent with tools and memory

        Args:
            groq_api_key: API key for Groq (if not in environment)
        """

        # Set Groq API key
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

        # Initialize Groq LLM
        self.llm = self._initialize_llm()

        # Load tools
        self.tools = get_tools()

        # Memory for conversation
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            k=10
        )

        # Prompt setup
        self.prompt = self._create_prompt()

        # Agent creation
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True,
           
            output_parser=SafeOutputParser()
        )

    def _initialize_llm(self):
        """Initialize Groq LLM"""
        
        try:
            # Get API key from environment
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            
            print("Initializing Groq LLM...")
            
            # Initialize ChatGroq with optimized settings for customer service
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama3-70b-8192",  # Fast and capable model
                temperature=0,  # Low temperature for consistent responses
                max_tokens=1024,
                top_p=0.9,
                streaming=False,
                # Additional parameters for better performance
                request_timeout=120,
                max_retries=10
            )
            
            print("✅ Groq LLM initialized successfully!")
            return llm
            
        except Exception as e:
            print(f"❌ Error initializing Groq LLM: {e}")
            print("Attempting fallback model...")
            return self._initialize_fallback_llm()

    def _initialize_fallback_llm(self):
        """Fallback to a different Groq model if the primary fails"""
        
        fallback_models = [
            "llama3-8b-8192",    # Smaller LLaMA 3 model
            "mixtral-8x7b-32768", # Mixtral model
            "gemma2-9b-it"        # Gemma model
        ]
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        for model_name in fallback_models:
            try:
                print(f"Trying fallback model: {model_name}")
                
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=model_name,
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=0.9,
                    streaming=False,
                    request_timeout=60,
                    max_retries=5
                )
                
                # Test the model with a simple query
                test_response = llm.invoke("Hello")
                
                print(f"✅ Fallback model {model_name} loaded successfully!")
                return llm
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        raise ValueError("Failed to load both primary and all fallback models")

    def _create_prompt(self):
        """Create the system prompt for the agent"""

        template ="""
You are an AI customer service representative for an e-commerce platform. Your role is to help customers with their inquiries in a friendly, professional, and efficient manner.

Your Capabilities:
- If the user greets with "Hello" or "Hi", respond with a friendly greeting.
- Handle customer inquiries about orders, products, and account information.
- Check order status, cancel orders, and process returns.
- Search for products and provide detailed product information.
- Access customer information and update preferences.
- Get weather information for shipping estimates.
- Provide product recommendations based on weather or preferences.
- Make autonomous decisions about which tools to use.
- Chain multiple tools together when needed.

Guidelines:
1. Be Proactive: Anticipate customer needs and offer relevant information.
2. Be Contextual: Remember previous conversation context and use it appropriately.
3. Be Autonomous: Decide which tools to use based on customer queries without asking for permission.
4. Be Helpful: If you can't solve a problem directly, offer alternatives or escalation paths.
5. Be Professional: Maintain a friendly, helpful tone while being efficient.
6. Chain Tools: Use multiple tools in sequence when it provides better customer service.

Important Notes:
- Always prioritize customer satisfaction.
- If unsure, ask for clarification rather than guessing.
- When handling cancellations or returns, explain the process clearly.
- Provide order IDs, product IDs, and reference numbers when relevant.
- Be empathetic with complaints or issues.
- Always include "Final Answer:" even if tools fail.
- Never leave the customer hanging without a response.
Always use the data from the database to answer the questions
if you cannot parse data from database alone give your best guess based on the data you have
Tools Available:
{tools}

Conversation history:
{chat_history}

Question: {input}
Thought: I should consider the user's request and decide the best tool to use.
Action: [Choose one from {tool_names}]
Action Input: [Input for the selected tool]
Observation: [Result from the tool]

[Repeat Thought/Action/Action Input/Observation steps if necessary, but limit repetitions.]

Thought: I now have the information I need to respond {agent_scratchpad}
Final Answer: [Your complete, helpful response to the user.]


"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

    def process_message(self, message: str, customer_context: Dict[str, Any] = None) -> str:
        """Process a customer message and return response"""

        try:
            enhanced_message = message
            if customer_context:
                context_info = f"\nCustomer Context: {customer_context}"
                enhanced_message = message + context_info

            chat_history = ""
            for msg in self.memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    chat_history += f"Human: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    chat_history += f"AI: {msg.content}\n"

            response = self.agent_executor.invoke({
                "input": enhanced_message,
                "chat_history": chat_history
            })

            self.memory.save_context(
                {"input": message},
                {"output": response["output"]}
            )

            return response["output"]

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            print(f"Error details: {traceback.format_exc()}")
            
            # Handle specific Groq API errors
            if "rate limit" in str(e).lower():
                error_msg = "I'm currently experiencing high demand. Please wait a moment and try again."
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                error_msg = "There seems to be an issue with the API authentication. Please contact support."
            elif "timeout" in str(e).lower():
                error_msg = "The request timed out. Please try again with a shorter message."
            
            self.memory.save_context(
                {"input": message},
                {"output": error_msg}
            )
            return error_msg

    def reset_conversation(self):
        """Reset the conversation memory"""
        self.memory.clear()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history

    def get_available_models(self) -> List[str]:
        """Get list of available Groq models"""
        return [
            "llama3-70b-8192",
            "llama3-8b-8192", 
           "gemma2-9b-it"
        ]

    def switch_model(self, model_name: str):
        """Switch to a different Groq model"""
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            new_llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=0.2,
                max_tokens=1024,
                top_p=0.9,
                streaming=False,
                request_timeout=60,
                max_retries=3
            )
            
            # Test the new model
            test_response = new_llm.invoke("Hello")
            
            # If successful, update the agent
            self.llm = new_llm
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                return_intermediate_steps=True
            )
            
            print(f"✅ Successfully switched to model: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to switch to model {model_name}: {e}")
            return False

    def cleanup(self):
        """Clean up resources (not needed for API-based models)"""
        print("🧹 Session cleaned up")


class CustomerContext:
    """Manage customer context and session information"""

    def __init__(self):
        self.sessions = {}

    def get_context(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def update_context(self, session_id: str, context: Dict[str, Any]):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id].update(context)

    def set_customer_id(self, session_id: str, customer_id: str):
        self.update_context(session_id, {"customer_id": customer_id})

    def set_customer_email(self, session_id: str, email: str):
        self.update_context(session_id, {"customer_email": email})


# Global customer context manager
customer_context_manager = CustomerContext()