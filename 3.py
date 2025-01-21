product_prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]

totalsum = 0

while True:
    product_number = input("Enter product number (1-10, 0 to quit): ")
    if product_number == "0":
        break
    try:
        product_number = int(product_number)
        if 1 <= product_number <= 10:
            price = product_prices[product_number - 1]
            totalsum += price
            print(f"Product number: {product_number}, Price: {price}")
        else:
            print("Invalid product number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

print(f"Total: {totalsum}")
payment = float(input("Payment: "))
change = payment - totalsum
print(f"Change: {change:.2f}")