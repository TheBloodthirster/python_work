def make_pizza(size,*toppings):
    print("\nMaking a "+ str(size) +
            "-omcj pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)