# NumPy

The most fundamental third-party package for scientific computing in Python is NumPy, which provides multidimensional **array** data types, along with a range of functions and methods to manipulate them. Other third-party packages, including Pandas (which we will look at next session), use NumPy arrays as backends for more specialized data structures.


While Python comes with several basic container types (`list`,`tuple`,`dict`), NumPy's arrays are much more **efficient** than the built-in types. This is particularly true for large data, for which NumPy scales much better than Python's built-in data structures.


## Basics of Numpy arrays

Once you have installed numpy, you can import it as


```python
import numpy
```

though we will use the conventional shorthand


```python
import numpy as np
```

As mentioned above, the main object provided by numpy is a powerful array.  We'll start by exploring how the numpy array differs from Python lists.  We will start by creating a simple list and an array with the same contents of the list:


```python
lst = list(range(1000))
arr = np.arange(1000)

# Here's what the array looks like
arr[:10]
```


```python
type(arr)
```


```python
%timeit [i**2 for i in lst]
```


```python
%timeit arr**2
```

Elements of a one-dimensional array are indexed with square brackets, as with lists:


```python
arr[5:10]
```


```python
arr[-1]
```

The first difference to note between lists and arrays is that arrays are **homogeneous**; i.e. all elements of an array must be of the same type.  Lists can contain elements of arbitrary type. For example, we can change the last element in our list above to be a string:


```python
lst[0] = 'a string inside a list'
lst[:10]
```

but the same can not be done with an array, as we get an error message:


```python
arr[0] = 'a string inside an array'
```

The information about the type of an array is contained in its *dtype* attribute:


```python
arr.dtype
```

Once an array has been created, its `dtype` is fixed and it can only store elements of the same type.  For this example where the dtype is integer, if we store a floating point number it will be automatically converted into an integer:


```python
arr[0] = 1.234
arr[:10]
```

Above we created an array from an existing list; now let us now see other ways in which we can create arrays, which we'll illustrate next.  A common need is to have an array initialized with a constant value, and very often this value is 0 or 1 (suitable as starting value for additive and multiplicative loops respectively); `zeros` creates arrays of all zeros, with any desired dtype:


```python
np.zeros(5, float)
```


```python
np.zeros(3, int)
```


```python
np.zeros(3, complex)
```

and similarly for `ones`:


```python
print('5 ones: {0}'.format(np.ones(5)))
```

If we want an array initialized with an arbitrary value, we can create an empty array and then use the fill method to put the value we want into the array:


```python
a = np.empty(4)
a
```


```python
a.fill(5.5)
a
```

The `arange` function generates an array for a range of integers. Similarly,  the `linspace` and `logspace` functions to create linearly and logarithmically-spaced **grids** respectively, with a fixed number of points and including both ends of the specified interval:


```python
np.linspace(0, 1, num=5)
```


```python
np.linspace(0, 1, endpoint=False, num=5)
```


```python
np.logspace(1, 4, num=4)
```

Finally, it is often useful to create arrays with random numbers that follow a specific **distribution**.  The `np.random` module contains a number of functions that can be used to this effect, for example this will produce an array of 5 random samples taken from a **standard normal** distribution (0 mean and variance 1):

$$f(x \mid \mu=0, \sigma=1) = \sqrt{\frac{1}{2\pi \sigma^2}} \exp\left\{ -\frac{x^2}{2\sigma^2} \right\}$$ 


```python
np.random.randn(5)
```

whereas the following will also give 5 samples, but from a normal distribution with a mean of 9 and a standard deviation of 3:


```python
norm10 = np.random.normal(loc=9, scale=3, size=10)
```

You can access the documentation for the `random` number generators, or any NumPy  function, using the `help` function.


```python
help(np.random.exponential)
```

More generally, you can search for NumPy help on a variety of topics, using the `lookfor` function. 


```python
np.lookfor('distribution')
```

## Exercise: Random numbers

Generate a NumPy array of 1000 random numbers sampled from a Poisson distribution, with parameter `lam=5`. What is the modal value in the sample?


```python
# Write your answer here
```

## Indexing with other arrays

Above we saw how to index arrays with single numbers and slices, just like Python lists.  But arrays allow for a more sophisticated kind of indexing which is very powerful: you can index an array with another array, and in particular with an array of boolean (`bool`) values.  This is particluarly useful to extract information from an array that matches a certain condition.

Consider for example that in the array `norm10` we want to replace all values above 9 with the value 0.  We can do so by first finding the *mask* that indicates where this condition is `True` or `False`:


```python
norm10
```


```python
mask = norm10 > 9
mask
```

Now that we have this mask, we can use it to return those values


```python
norm10[mask]
```

or to change their values


```python
norm10[mask] = 0
norm10
```

## Multidimensional Arrays

NumPy can create arrays of aribtrary dimensions, and all the methods illustrated in the previous section work with more than one dimension. For example, a list of lists can be used to initialize a two dimensional array:


```python
samples_list = [[632, 1638, 569, 115], [433,1130,754,555]]
samples_array = np.array(samples_list)
samples_array.shape
```

With two-dimensional arrays we start seeing the convenience of NumPy data structures: while a nested list can be indexed across dimensions using consecutive `[ ]` operators, multidimensional arrays support a more natural indexing syntax with a single set of brackets and a set of comma-separated indices:


```python
samples_list[0][1]
```


```python
samples_array[0,1]
```

Most of the array creation functions listed above can be passed multidimensional shapes. For example:


```python
np.zeros((2,3))
```


```python
np.random.normal(10, 3, size=(2, 4))
```

In fact, an array can be **reshaped** at any time, as long as the total number of elements is unchanged.  For example, if we want a 2x4 array with numbers increasing from 0, the easiest way to create it is via the array's `reshape` method.


```python
arr = np.arange(8).reshape(2,4)
arr
```

With multidimensional arrays, you can also use slices, and you can mix and match slices and single indices in the different dimensions (using the same array as above):


```python
arr[1, 2:4]
```


```python
arr[:, 2]
```

If you only provide one index, then you will get the corresponding row.


```python
arr[1]
```

Now that we have seen how to create arrays with more than one dimension, it's a good idea to look at some of the most useful **properties and methods** that arrays have.  The following provide basic information about the size, shape and data in the array:


```python
print('Data type                :', samples_array.dtype)
print('Total number of elements :', samples_array.size)
print('Number of dimensions     :', samples_array.ndim)
print('Shape (dimensionality)   :', samples_array.shape)
print('Memory used (in bytes)   :', samples_array.nbytes)
```

Arrays also have many useful methods, some especially useful ones are:


```python
print('Minimum and maximum             :', samples_array.min(), samples_array.max())
print('Sum, mean and standard deviation:', samples_array.sum(), samples_array.mean(), samples_array.std())
```

For these methods, the above operations area all computed on all the elements of the array.  But for a multidimensional array, it's possible to do the computation along a single dimension, by passing the `axis` parameter; for example:


```python
samples_array.sum(axis=0)
```


```python
samples_array.sum(axis=1)
```

As you can see in this example, the value of the `axis` parameter is the dimension which will be *consumed* once the operation has been carried out.  This is why to sum along the rows we use `axis=0`.  

This can be easily illustrated with an example that has more dimensions; we create an array with 4 dimensions and shape `(3,4,5,6)` and sum along the axis index 2.  That consumes the dimension whose length was 5, leaving us with a new array that has shape `(3,4,6)`:


```python
np.zeros((3,4,5,6)).sum(2).shape
```

Another widely used property of arrays is the `.T` attribute, which allows you to access the transpose of the array:


```python
samples_array.T
```

Which is the equivalent of calling NumPy's `transpose` function:


```python
np.transpose(samples_array)
```

There is a wide variety of methods and properties of arrays.       


```python
[attr for attr in dir(samples_array) if not attr.startswith('__')]
```

### Exercises: Matrix Creation

Generate the following structure as a numpy array, without typing the values by hand. Then, create another array containing just the 2nd and 4th rows.

        [[1,  6, 11],
         [2,  7, 12],
         [3,  8, 13],
         [4,  9, 14],
         [5, 10, 15]]


```python
# Write your answer here
```

Create a **tridiagonal** matrix with 5 rows and columns, with 1's on the diagonal and 2's on the off-diagonal.


```python
# Write your answer here
```

## Array Operations

Arrays support all regular arithmetic operators, and NumPy also contains a complete collection of basic mathematical functions that operate on arrays.  It is important to remember that in general, all operations with arrays are applied **element-wise**, that is, applied to each element of the array.  

Consider for example:


```python
sample1, sample2 = np.array([632, 1638, 569, 115]), np.array([433,1130,754,555])
sample_sum = sample1 + sample2

print('{0} + {1} = {2}'.format(sample1, sample2, sample_sum))
```

This includes the multiplication operator -- it does *not* perform matrix multiplication, as is the case in Matlab, for example:


```python
print('{0} X {1} = {2}'.format(sample1, sample2, sample1*sample2))
```

While this implies that the dimension of the arrays for each operation must match in size, numpy will **broadcast** dimensions when possible.  For example, suppose that you want to add the number 1.5 to each element `arr1`. One approach is to use the `ones` function to match the dimension of the array.


```python
sample1 + 1.5*np.ones(4)
```

But thanks to numpy's broadcasting rules, the following is equally valid:


```python
sample1 + 1.5
```

In this case, numpy looked at both operands and saw that the first was a one-dimensional array of length 4 and the second was a scalar, considered a zero-dimensional object. The broadcasting rules allow numpy to:

* *create* new array of length 1 
* *extend* the array to match the size of the corresponding array

So in the above example, the scalar 1.5 is effectively cast to a 1-dimensional array of length 1, then stretched to length 4 to match the dimension of `arr1`. After this, element-wise addition can proceed as now both operands are one-dimensional arrays of length 4.

This broadcasting behavior is powerful, especially because when NumPy broadcasts to create new dimensions or to stretch existing ones, it doesn't actually replicate the data.  In the example above the operation is carried *as if* the 1.5 was a 1-d array with 1.5 in all of its entries, but no actual array was ever created.  This saves memory and improves the **performance** of operations.

When broadcasting, NumPy compares the sizes of each dimension in each operand. It starts with the trailing dimensions, working forward and creating dimensions as needed to accomodate the operation. Two dimensions are considered compatible for operation when:

* they are equal in size
* one is scalar (or size 1)

If these conditions are not met, an exception is thrown, indicating that the arrays have incompatible shapes. 


```python
sample1 + np.array([7,8])
```

Let's create a 1-dimensional array and add it to a 2-dimensional array, to illustrate broadcasting:


```python
b = np.array([10, 20, 30, 40])
bcast_sum = sample1 + b

print('{0}\n\n+ {1}\n{2}\n{3}'.format(sample1, b, '-'*21, bcast_sum))
```

What if we wanted to add `[-100, 100]` to the rows of `sample1`?  Direct addition will not work:


```python
c = np.array([-100, 100])
sample1 + c
```

Remember that matching begins at the **trailing** dimensions. Here, `c` would need to have a trailing dimension of 1 for the broadcasting to work.  We can augment arrays with dimensions on the fly, by indexing it with a `np.newaxis` object, which adds an "empty" dimension:


```python
cplus = c[:, np.newaxis]
cplus
```

This is exactly what we need, and indeed it works:


```python
sample1 + cplus
```

For the full broadcasting rules, please see the official Numpy docs, which describe them in detail and with more complex examples.

### Exercises: Array manipulation

Divide each column of the array:

        np.array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

elementwise with the array `np.array([1., 5, 10, 15, 20])`.


```python
# Write your answer here
```

Generate a 10 x 3 array of random numbers (in range [0,1]). For each row, pick the number closest to 0.5.

*Hints*:

* Use `abs` and `argsort` to find the column `j` closest for each row.
* Use "fancy" indexing to extract the numbers.



```python
# Write your answer here
```

## Linear Algebra

Numpy includes a linear algebra submodule, along with a suite of `array` methods for performing linear algebra. For example, the `dot` method performs an inner (dot) product on vectors and matrices:


```python
v1 = np.array([2, 3, 4])
v2 = np.array([1, 0, 1])

v1.dot(v2)
```

Equivalently, we can use the `dot` function:


```python
np.dot(v1, v2)
```

When performing regular matrix-vector multiplication, note that NumPy makes no distinction between row and column vectors *per se* and simply verifies that the dimensions match the required rules of matrix multiplication, in this case we have a $2 \times 3$ matrix multiplied by a 3-vector, which produces a 2-vector:


```python
A = np.arange(6).reshape(2, 3)

A.dot(v1)
```

For matrix-matrix multiplication, the same dimension-matching rules must be satisfied, e.g. consider the difference between $A \times A^T$:


```python
A.dot(A.T)
```

and $A^T \times A$:


```python
A.T.dot(A)
```

Beyond inner products, the `numpy.linalg` module includes functions for calculating determinants, matrix norms, Cholesky decomposition, eigenvalue and singular value decompositions, and more.  

Additional linear algebra tools are available in SciPy's linear algebra library, `scipy.linalg`. It includes the majority of the tools in the classic LAPACK libraries as well as functions to operate on sparse matrices.  

## Reading and writing data

NumPy lets you save and retrive data structures to and from files on a local or remote storage, in either **text** or **binary** formats. Which format is appropriate depends on which tradeoff that you are willing to make:

* **Text mode**: occupies more space, precision can be lost (if not all digits are written to disk), but is readable and editable by hand with a text editor.  Storage is limited to one- and two-dimensional arrays.

* **Binary mode**: compact and exact representation of the data in memory, can't be read or edited by hand.  Arrays of any size and dimensionality can be saved and read without loss of information.

First, let's see how to read and write arrays in text mode.  The `np.savetxt` function saves an array to a text file, with options to control the precision, separators and even adding a header:


```python
arr = np.arange(10).reshape(2, 5)
np.savetxt('test.out', arr, fmt='%.2e', header="My dataset")
```


```python
!cat test.out
```

And this same type of file can then be read with the matching `np.loadtxt` function:


```python
arr2 = np.loadtxt('test.out')
arr2
```

For binary data, we use either `np.save` or `np.savez`.  The first saves a single array to a file with `.npy` extension, while the latter can be used to save a *group* of arrays into a single file with `.npz` extension.  The files created with these routines can then be read with the `np.load` function.

Let us first see how to use the simpler `np.save` function to save a single array:


```python
np.save('test.npy', arr2)
```

This can be read back:


```python
arr2n = np.load('test.npy')
```

And we can confirm that they are equal:


```python
np.any(arr2 - arr2n)
```

Now let us see how the `np.savez` function works.  

It expects both a filename and either a sequence of arrays or a set of key-value pairs.  If arrays are passed, the `savez` will automatically name the saved arrays in the archive as `arr_0`, `arr_1`, ...


```python
np.savez('test.npz', arr, arr2)
arrays = np.load('test.npz')
arrays.files
```

Alternatively, if we explicitly name the arrays using keyword arguments:


```python
np.savez('test.npz', foo=arr, bar=arr2)
arrays = np.load('test.npz')
arrays.files
```

The object returned by `np.load` from an `.npz` file works like a dictionary, though you can also access its constituent files by attribute using its special `.f` field; this is best illustrated with an example with the `arrays` object from above:


```python
# First row of array
arrays['bar'][0]
```

Equivalently:


```python
arrays.f.bar[0]
```

This `.npz` format is a very convenient way to package compactly and without loss of information, into a single file, a group of related arrays that pertain to a specific problem.  At some point, however, the complexity of your dataset may be such that the optimal approach is to use one of the standard formats in scientific data processing that have been designed to handle complex datasets, such as NetCDF or HDF5.  

## Guided Exercise: Structured Arrays 

Import the `microbiome.csv` dataset in the `data/microbiome` directory using NumPy's `loadtxt` function. This will take some experimentation; use the built-in help to get hints!


```python
# Write answer here
```
