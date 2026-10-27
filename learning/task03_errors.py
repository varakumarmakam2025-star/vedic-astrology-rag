def safe_divide(a, b):
    try:
        result = a / b
        print(f"Result: {result}")

    except ZeroDivisionError:
        print("Can't divide by zero!")

safe_divide(10, 2)
safe_divide(10, 0)   

