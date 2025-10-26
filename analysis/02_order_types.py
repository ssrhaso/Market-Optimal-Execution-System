#Simulates how interaction with market happens using different order type commands

# Order class to represent a stock order
class Order:
    #Initialise order with side, quantity, type, and optional price
    def __init__(self, side, quantity, order_type, price=None):
        self.side = side                    # 'buy' or 'sell'
        self.quantity = quantity            # number of shares
        self.order_type = order_type        # 'market' or 'limit'
        self.price = price                  # limit price for limit orders

    # String representation of the Order
    def __str__(self):
        if self.order_type == 'market':
            return f"{self.side.capitalize()} {self.quantity} shares @ market price"
        else:
            return f"{self.side.capitalize()} {self.quantity} shares @ limit {self.price}"
    
    
# Function to take user input and create an Order 
def input_order(side, quantity, order_type, price=None):
    side = input("Enter order side (buy/sell): ").strip().lower()
    quantity = int(input("Enter quantity: "))
    order_type = input("Enter order type (market/limit): ").strip().lower()
    if order_type == "limit":
        price = float(input("Enter limit price: "))
    
    print(Order(side, quantity, order_type, price))

        
        
if __name__ == "__main__":
    # Test the Order class
    side = ''
    quantity = 0
    order_type = ''
    price = None
    input_order(side, quantity, order_type, price)
    