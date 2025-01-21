def my_split(sentence, separator):
    result = []
    temp = ""
    for char in sentence:
        if char == separator:
            result.append(temp)
            temp = ""
        else:
            temp += char
    result.append(temp)  # Add the last segment
    return result

def my_join(items, separator):
    result = ""
    for i, item in enumerate(items):
        if i > 0:
            result += separator
        result += item
    return result

sentence = input("Please enter sentence: ")
split_result = my_split(sentence, " ")

joined_sentence = my_join(split_result, ",")
print(joined_sentence)

for word in split_result:
    print(word)