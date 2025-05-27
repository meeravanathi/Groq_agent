from mockdatabase import MockOrderDatabase, MockProductDatabase, MockCustomerDatabase

def serialize_product_data():
    db = MockProductDatabase()
    documents = []
    for product in db.products.values():
        text = f"""Product Name: {product['name']}
Category: {product['category']}
Price: ${product['price']}
Availability: {product['availability']}
Description: {product['description']}
Rating: {product['rating']}
Features: {', '.join(product['features'])}"""
        documents.append({"id": product["product_id"], "text": text})
    return documents

def serialize_customer_data():
    db = MockCustomerDatabase()
    documents = []
    for customer in db.customers.values():
        text = f"""Customer Name: {customer['name']}
Email: {customer['email']}
Phone: {customer['phone']}
Address: {customer['address']}
Loyalty Tier: {customer['tier']}
Preferences: Categories - {', '.join(customer['preferences']['categories'])}; Brands - {', '.join(customer['preferences']['brands'])}"""
        documents.append({"id": customer["customer_id"], "text": text})
    return documents

def serialize_order_data():
    db = MockOrderDatabase()
    documents = []
    for order in db.orders.values():
        items = ', '.join([f"{item['quantity']}x {item['name']}" for item in order["items"]])
        text = f"""Order ID: {order['order_id']}
Customer ID: {order['customer_id']}
Status: {order['status']}
Items: {items}
Total: ${order['total']}
Shipping Address: {order['shipping_address']}
Tracking: {order.get('tracking_number', 'None')}"""
        documents.append({"id": order["order_id"], "text": text})
    return documents
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

documents = serialize_product_data() + serialize_customer_data() + serialize_order_data()
texts = [doc["text"] for doc in documents]
metadata = [{"id": doc["id"]} for doc in documents]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadata)
vectorstore.save_local("vectorstore/products")
