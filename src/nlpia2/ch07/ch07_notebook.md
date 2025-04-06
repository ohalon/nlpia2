```python
kernel = [1, 1]
inpt = [1, 2, 0, -1, -2, 0, 1, 2]

output = []
for i in range(len(inpt) - 1):  # <1>
    output.append(
        sum(
            [
                inpt[i + k] * kernel[k]
                for k in range(2)  # <2>
            ]
        )
    )

    
# <1> Stop at second to last input value so the window size of 2 doesn't slide off the end of the input
# <2> kernel is 2 long and the list comprehension iterates over the kernel length

output
```




    [3, 2, -1, -3, -2, 1, 3]



`[3, 2, -1, -3, -2, 1, 3]`


```python
def convolve(inpt, kernel):
    output = []
    for i in range(len(inpt) - len(kernel) + 1):
        output.append(
            sum(
                [
                    inpt[i + k] * kernel[k]
                    for k in range(len(kernel))
                ]
            )
        )
    return output

convolve(inpt=inpt, kernel=[1, 1, 1])
```




    [3, 1, -3, -3, -1, 3]



`[3, 1, -3, -3, -1, 3]`
