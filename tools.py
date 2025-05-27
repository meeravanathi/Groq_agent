"""
Tool implementations for the e-commerce chatbot
"""
from typing import Type
import json
import requests
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from mock_databases import MockOrderDatabase, MockProductDatabase, MockCustomerDatabase
import os
from datetime import datetime

# Initialize mock databases
order_db = MockOrderDatabase()
product_db = MockProductDatabase()
customer_db = MockCustomerDatabase()

class OrderStatusInput(BaseModel):
    order_id: str = Field(description="The order ID to check status for")

class OrderCancelInput(BaseModel):
    order_id: str = Field(description="The order ID to cancel")

class ReturnProcessInput(BaseModel):
    order_id: str = Field(description="The order ID to process return for")
    reason: str = Field(description="Reason for return", default="")

class ProductSearchInput(BaseModel):
    query: str = Field(description="Search query for products")
    category: Optional[str] = Field(description="Optional category filter", default=None)

class ProductDetailsInput(BaseModel):
    product_id: str = Field(description="Product ID to get details for")

class CustomerInfoInput(BaseModel):
    customer_id: Optional[str] = Field(description="Customer ID", default=None)
    email: Optional[str] = Field(description="Customer email", default=None)

class CustomerOrdersInput(BaseModel):
    customer_id: str = Field(description="Customer ID to get orders for")

class SearchOrdersByEmailInput(BaseModel):
    email: str = Field(description="Customer email to search orders")

class UpdatePreferencesInput(BaseModel):
    customer_id: str = Field(description="Customer ID")
    preferences: Dict[str, Any] = Field(description="Preferences to update")

class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for")

class RecommendationInput(BaseModel):
    category: Optional[str] = Field(description="Product category", default=None)
    weather_condition: Optional[str] = Field(description="Current weather condition", default=None)

class OrderStatusTool(BaseTool):
    name: str = "order_status"
    description: str = "Check the status of an order by order ID. Use this when customers ask about their order status, tracking, or delivery information."
    args_schema: Type[BaseModel] = OrderStatusInput

    def _run(self, order_id: str) -> str:
        order = order_db.get_order_status(order_id)
        if not order:
            return f"RESULT: Order {order_id} not found. This order ID does not exist in our system. Please verify the order ID or ask the customer for their email to search for orders differently."
        
        status_info = f"""
FINAL ANSWER: Order Details Found
Order ID: {order['order_id']}
Status: {order['status'].title()}
Order Date: {order['order_date']}
Total: ${order['total']:.2f}
Items:
"""
        for item in order['items']:
            status_info += f"  - {item['name']} (Qty: {item['quantity']}) - ${item['price']:.2f}\n"
        
        status_info += f"Shipping Address: {order['shipping_address']}\n"
        
        if order['tracking_number']:
            status_info += f"Tracking Number: {order['tracking_number']}\n"
        
        return status_info.strip()

class OrderCancelTool(BaseTool):
    name: str = "cancel_order"
    description: str = "Cancel an order if it's still possible. Use this when customers want to cancel their orders."
    args_schema: Type[BaseModel] = OrderCancelInput
    
    def _run(self, order_id: str) -> str:
        result = order_db.cancel_order(order_id)
        return f"RESULT: {result['message']}"

class ReturnProcessTool(BaseTool):
    name: str = "process_return"
    description: str = "Process a return request for a delivered order. Use this when customers want to return items."
    args_schema: Type[BaseModel] = ReturnProcessInput
    
    def _run(self, order_id: str, reason: str = "") -> str:
        result = order_db.process_return(order_id, reason)
        return f"RESULT: {result['message']}"

class ProductSearchTool(BaseTool):
    name: str = "search_products"
    description: str = "Search for products by name or category. Use this when customers are looking for specific products or browsing categories."
    args_schema: Type[BaseModel] = ProductSearchInput
    
    def _run(self, query: str, category: str = None) -> str:
        products = product_db.search_products(query, category)
        
        if not products:
            return f"RESULT: No products found for '{query}'" + (f" in category '{category}'" if category else "") + ". You might want to try different search terms or browse our categories."
        
        result = f"RESULT: Found {len(products)} product(s):\n\n"
        for product in products:
            result += f"**{product['name']}** (ID: {product['product_id']})\n"
            result += f"Category: {product['category']}\n"
            result += f"Price: ${product['price']:.2f}\n"
            result += f"Availability: {product['availability'].replace('_', ' ').title()}\n"
            result += f"Rating: {product['rating']}/5.0\n"
            result += f"Description: {product['description']}\n\n"
        
        return result.strip()

class ProductDetailsTool(BaseTool):
    name: str = "product_details"
    description: str = "Get detailed information about a specific product by product ID. Use this when customers need detailed product information or asks for product search."
    args_schema: Type[BaseModel] = ProductDetailsInput
    
    def _run(self, product_id: str) -> str:
        product = product_db.get_product_details(product_id)
        
        if not product:
            return f"RESULT: Product {product_id} not found. This product ID does not exist in our catalog."
        
        details = f"""
RESULT: Product Details Found
**{product['name']}** (ID: {product['product_id']})
Category: {product['category']}
Price: ${product['price']:.2f}
Availability: {product['availability'].replace('_', ' ').title()}
Stock: {product['stock_count']} units
Rating: {product['rating']}/5.0
Description: {product['description']}

Features:
"""
        for feature in product['features']:
            details += f"  • {feature}\n"
        
        return details.strip()

class CustomerInfoTool(BaseTool):
    name: str = "customer_info"
    description: str = "Get customer information by customer ID or email.Use this when customers ask about their account details, preferences, or loyalty points or forgets their customer ID."
    args_schema: Type[BaseModel] = CustomerInfoInput
    
    def _run(self, customer_id: str = None, email: str = None) -> str:
        if customer_id:
            customer = customer_db.get_customer_info(customer_id)
        elif email:
            customer = customer_db.get_customer_by_email(email)
        else:
            return "RESULT: Error - Please provide either customer ID or email address."
        
        if not customer:
            if customer_id:
                return f"RESULT: Customer ID {customer_id} not found in our system. This customer ID does not exist. Try asking the customer for their email address instead or use the get_customer_orders tool to search for their orders directly."
            else:
                return f"RESULT: No customer found with email {email}. This email is not registered in our system."
        
        info = f"""
RESULT: Customer Information Found
Name: {customer['name']}
Email: {customer['email']}
Phone: {customer['phone']}
Address: {customer['address']}
Loyalty Points: {customer['loyalty_points']}
Tier: {customer['tier']}

Preferences:
  Categories: {', '.join(customer['preferences']['categories'])}
  Brands: {', '.join(customer['preferences']['brands'])}
  Communication: {customer['preferences']['communication']}

Recent Orders: {', '.join(customer['order_history'])}
"""
        return info.strip()

class CustomerOrdersTool(BaseTool):
    name: str = "get_customer_orders"
    description: str = "Get all orders for a customer by customer ID. Use this when customer forgets their order IDs or wants to see all their orders."
    args_schema: Type[BaseModel] = CustomerOrdersInput
    
    def _run(self, customer_id: str) -> str:
        # Try to get customer first to verify they exist
        customer = customer_db.get_customer_info(customer_id)
        if not customer:
            return f"RESULT: Customer ID {customer_id} not found. Cannot retrieve orders for non-existent customer. Ask customer for email address to search alternatively."
        
        # Get customer's order history
        orders = customer.get('order_history', [])
        if not orders:
            return f"RESULT: No orders found for customer {customer_id} ({customer['name']}). This customer has not placed any orders yet."
        
        result = f"RESULT: Orders found for {customer['name']} (ID: {customer_id}):\n\n"
        for order_id in orders:
            order = order_db.get_order_status(order_id)
            if order:
                result += f"Order {order_id}: {order['status'].title()} - ${order['total']:.2f} (Date: {order['order_date']})\n"
            else:
                result += f"Order {order_id}: Status unknown\n"
        
        result += f"\nTotal orders: {len(orders)}"
        return result

class SearchOrdersByEmailTool(BaseTool):
    name: str = "search_orders_by_email"
    description: str = "Search for orders using customer email when customer ID lookup fails. Alternative way to find customer orders."
    args_schema: Type[BaseModel] = SearchOrdersByEmailInput
    
    def _run(self, email: str) -> str:
        # First try to find customer by email
        customer = customer_db.get_customer_by_email(email)
        if not customer:
            return f"RESULT: No customer found with email {email}. This email is not registered in our system."
        
        # Get their orders
        orders = customer.get('order_history', [])
        if not orders:
            return f"RESULT: Customer with email {email} exists but has no orders yet."
        
        result = f"RESULT: Orders found for {email}:\n\n"
        for order_id in orders:
            order = order_db.get_order_status(order_id)
            if order:
                result += f"Order {order_id}: {order['status'].title()} - ${order['total']:.2f} (Date: {order['order_date']})\n"
        
        result += f"\nTotal orders: {len(orders)}"
        return result

class UpdatePreferencesTool(BaseTool):
    name: str = "update_preferences"
    description: str = "Update customer preferences such as preferred categories, brands, or communication methods."
    args_schema: Type[BaseModel] = UpdatePreferencesInput
    
    def _run(self, customer_id: str, preferences: Dict[str, Any]) -> str:
        result = customer_db.update_preferences(customer_id, preferences)
        return f"RESULT: {result['message']}"

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather information for a city. Use this for shipping estimates or weather-based product recommendations."
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str) -> str:
        api_key = os.getenv("WEATHER_API_KEY")
        
        if not api_key:
            print("⚠️ Weather API key not configured. Using mock data.")
            weather_conditions = ["sunny", "rainy", "cloudy", "snowy", "windy"]
            condition = weather_conditions[hash(city) % len(weather_conditions)]
            temp = 15 + (hash(city) % 30)
            return f"RESULT: Weather in {city}: {condition.title()}, {temp}°C. " \
                   f"{'Good conditions for shipping.' if condition in ['sunny', 'cloudy'] else 'Potential shipping delays due to weather.'}"
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if response.status_code == 200:
                weather = data['weather'][0]['description']
                temp = data['main']['temp']
                return f"RESULT: Weather in {city}: {weather.title()}, {temp}°C. " \
                       f"{'Good conditions for shipping.' if 'clear' in weather or 'cloud' in weather else 'Potential shipping delays due to weather.'}"
            else:
                return f"RESULT: Could not get weather information for {city}."
        except Exception as e:
            return f"RESULT: Error getting weather information: {str(e)}"

class ProductRecommendationTool(BaseTool):
    name: str = "product_recommendations"
    description: str = "Get product recommendations based on category or weather conditions. Use this to suggest products to customers."
    args_schema: Type[BaseModel] = RecommendationInput
    
    def _run(self, category: str = None, weather_condition: str = None) -> str:
        recommendations = product_db.get_recommendations(category, weather_condition)
        
        if not recommendations:
            return "RESULT: No recommendations available at the moment."
        
        result = "RESULT: Here are some recommended products:\n\n"
        for product in recommendations[:5]:
            result += f"**{product['name']}** - ${product['price']:.2f}\n"
            result += f"{product['description']}\n"
            result += f"Rating: {product['rating']}/5.0\n\n"
        
        return result.strip()
class AskQueryTool(BaseTool):
    name: str = "ask_query"
    description: str = "Fallback tool to ask the customer for more details or clarify their request."
    args_schema: Type[BaseModel] = BaseModel  # no args needed

    def _run(self, **kwargs) -> str:
        return "RESULT: Could you please provide more details or clarify your request so I can assist you better?"

# List of all tools
def get_tools():
    return [
        OrderStatusTool(),
        OrderCancelTool(),
        ReturnProcessTool(),
        ProductSearchTool(),
        ProductDetailsTool(),
        CustomerInfoTool(),
        CustomerOrdersTool(),  # New tool for getting customer orders
        SearchOrdersByEmailTool(),  # New tool for email-based order search
        UpdatePreferencesTool(),
        WeatherTool(),
        ProductRecommendationTool(),
        AskQueryTool()
    ]