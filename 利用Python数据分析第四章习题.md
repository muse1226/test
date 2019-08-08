

```python
import numpy as np
```


```python
print(np.__version__)
np.show_config()
```

    1.16.4
    mkl_info:
        libraries = ['mkl_rt']
        library_dirs = ['G:/anaconda\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'G:/anaconda\\Library\\include']
    blas_mkl_info:
        libraries = ['mkl_rt']
        library_dirs = ['G:/anaconda\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'G:/anaconda\\Library\\include']
    blas_opt_info:
        libraries = ['mkl_rt']
        library_dirs = ['G:/anaconda\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'G:/anaconda\\Library\\include']
    lapack_mkl_info:
        libraries = ['mkl_rt']
        library_dirs = ['G:/anaconda\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'G:/anaconda\\Library\\include']
    lapack_opt_info:
        libraries = ['mkl_rt']
        library_dirs = ['G:/anaconda\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'G:/anaconda\\Library\\include']
    

##  3.创建一个大小为10的空向量


```python
print(np.empty(10))
print(np.zeros(10))
print(np.full((2,3),5.0))# np.full 构造一个大小为 shape 的用指定值填满的数组,
```

    [6.23042070e-307 7.56587584e-307 1.37961302e-306 6.23053614e-307
     6.23053954e-307 9.34609790e-307 8.45593934e-307 9.34600963e-307
     9.34603000e-307 3.39985345e-317]
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [[5. 5. 5.]
     [5. 5. 5.]]
    

## 4.查看数组占用内存大小


```python
sample4_1 = np.empty((3, 2), np.uint32)# uint32 无符号整数（0到42949967295））
sample4_2 = np.empty((3, 2), np.float16)# float 半精度浮点
print(sample4_1.itemsize * sample4_1.size)
print(sample4_2.itemsize * sample4_2.size)
```

    24
    12
    

## 5.查看numpy中add函数的用法


```python
np.info(np.add)
```

    add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
    Add arguments element-wise.
    
    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.  If ``x1.shape != x2.shape``, they must be
        broadcastable to a common shape (which may be the shape of one or
        the other).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.
    
    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.
    
    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])
    

## 6.创建一个大小为10的空向量，将第5个值设为1


```python
x=np.zeros(10)
x[4]=1
print(x)
```

    [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
    

## 7.用10到49的序列构建一个向量


```python
x=np.arange(10,49,2)
x
```




    array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
           44, 46, 48])



## 8.将一个数组变换倒序（最后一个元素成为第一个元素）


```python
print(np.arange(10))
print(np.arange(10)[::-1])# hint  这里是python的切片[起：止：间隔]
print(np.arange(0,10,2))
```

    [0 1 2 3 4 5 6 7 8 9]
    [9 8 7 6 5 4 3 2 1 0]
    [0 2 4 6 8]
    

## 9.用0-8这9个数构造一个3x3大小的矩阵


```python
x=np.arange(9).reshape((3,3))
x

```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])



## 10.从数组[1,2,0,0,4,0]中找出非0元素的下标


```python
print(np.nonzero([1,2,0,0,4,0]))
```

    (array([0, 1, 4], dtype=int64),)
    

## 11.创建3x3的对角矩阵


```python
print(np.identity(3))

# identity 只能创建方阵，eye要灵活一些，可以创建NxM的矩阵，也可以控制对角线的位置
print(np.eye(3,3,0))
#默认第一个和第二个参数相等，第三个参数为对角线位置
```

    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    

## 12.用随机数创建一个3x3x3的矩阵


```python
print(np.random.random())#默认生成0到1随机数，random模块下的random方法
print(np.random.random((3,3,3)))
```

    0.047990319234644474
    [[[0.10228906 0.6351556  0.36386859]
      [0.77053388 0.552616   0.43292743]
      [0.96736534 0.57974775 0.0597077 ]]
    
     [[0.58108979 0.31006738 0.74341004]
      [0.27774591 0.67038409 0.57007925]
      [0.82281656 0.38876254 0.69476672]]
    
     [[0.49429043 0.8784641  0.17820651]
      [0.37645734 0.64196145 0.18862474]
      [0.56931961 0.5982119  0.2124895 ]]]
    

## 13.创建一个10x10的随机数矩阵，并找到最大值和最小值


```python
sample13 = np.random.random((10, 10))
print(sample13)
print(sample13.min(), np.max(sample13))
```

    [[0.65062662 0.33029046 0.7505139  0.77434544 0.80682994 0.21255943
      0.77802965 0.3028319  0.93185456 0.34159609]
     [0.12903757 0.06230981 0.76344013 0.14266298 0.2459246  0.63423176
      0.43614371 0.96656003 0.90629995 0.05217631]
     [0.18851757 0.80251292 0.12761286 0.57429373 0.9381224  0.47411464
      0.0778607  0.52036847 0.27269696 0.19896503]
     [0.24028162 0.40591769 0.95056137 0.62799869 0.23476234 0.4141502
      0.57433578 0.27818752 0.938084   0.63262705]
     [0.10799752 0.01031902 0.31151009 0.93743366 0.02171799 0.43306017
      0.61057051 0.55108209 0.24668242 0.67713748]
     [0.95237335 0.4183695  0.83543674 0.09709669 0.87655023 0.72012478
      0.80727659 0.93786141 0.4687439  0.15134775]
     [0.22649374 0.60176198 0.77233949 0.00226468 0.53194804 0.12620741
      0.1363247  0.56325222 0.61599564 0.82620965]
     [0.02903684 0.35070549 0.83036838 0.96345582 0.55737373 0.07254831
      0.76321452 0.42563936 0.58993589 0.2145265 ]
     [0.30536335 0.39753383 0.75385136 0.97474638 0.0412178  0.65161627
      0.18781521 0.30762829 0.91161571 0.70293599]
     [0.99182063 0.88727612 0.69616307 0.68544553 0.16906266 0.51103184
      0.97560982 0.40719508 0.51362066 0.95573197]]
    0.0022646771886751793 0.991820634759905
    

## 14.创建一个大小为30的数组，并计算其算术平均值


```python
# mean计算算术平均值 average 计算加权平均值np.average(np.arange(1, 11) , weights=np.arange(10, 0, -1))
print(np.random.random(30).mean())
```

    0.4217053297781374
    

## 15. 创建一个二维数组，边为1，其余为0 


```python
sample15 = np.ones((5, 5))
print(sample15)
sample15[1:-1, 1:-1] = 0
print(sample15)
```

    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    [[1. 1. 1. 1. 1.]
     [1. 0. 0. 0. 1.]
     [1. 0. 0. 0. 1.]
     [1. 0. 0. 0. 1.]
     [1. 1. 1. 1. 1.]]
    


```python

```
