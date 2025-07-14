def sub(a, b):
    """Subtracts two numbers."""
    return a - b
    
def add(a, b):
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    result = add(3, 5)
    print(f"The result of adding 3 and 5 is: {result}")
    
    num1 = 10
    num2 = 5
    result = sub(num1, num2)
    print(f"The result of subtracting {num2} from {num1} is: {result}")
