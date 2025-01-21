shopping_list = []

while True:
    action = input("Would you like to: \n (1) Add or\n (2) Remove items or\n (3) Quit?: ")
    if action == "1":
        shopping_list.append(input("What will be added?: "))
    elif action == "2":
        if shopping_list:
            print(f"There are {len(shopping_list)} items in the list.")
            try:
                index = int(input("Which item is deleted?: ")) - 1
                if 0 <= index < len(shopping_list):
                    shopping_list.pop(index)
                else:
                    print("Incorrect selection.")
            except ValueError:
                print("Incorrect selection.")
        else:
            print("The list is empty, nothing to remove.")
    elif action == "3":
        print("The following items remain in the list:")
        for item in shopping_list:
            print(item)
        break
    else:
        print("Incorrect selection.")
