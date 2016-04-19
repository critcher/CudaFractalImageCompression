#Compressed File Format
This document holds a specification for the compressed format we would like to store files with. 

The table below shows the various values needed to be saved within the compressed file.

| Variable        | Type          |  Description  |
| :------------- |:-------------| :-----|
|width|int| width of the original image in pixels
|height|int| height of the original image in pixels
|rangeSize|int| the length of one side of a range block in pixels (range blocks are square and their size is fixed)
|domainSize|int| the lenght of one side of a domain block in pixels (domain blocks are square and their size is fixed) 
|domainX|int| the X coordinate of the top left corner of the domain block we are mapping from
|domainY|int| the Y coordinate of the top left corner of the domain block we are mapping from
|transformationNum|enum|, the transformation used on the domain block (one of 8 possible options) 
|brightnessOffset|[-255,255]| how bright the range block should be as compared to the domain block
|contrastFactor|float|how much the contrast changes from the orginal domain block

##Example File Format:
(using variables described above)

```
<width> <height>
<rangeSize> <domainSize>
<domainX> <domainY> <transformationNum> <brightnessOffset> <contrastFactor>
<domainX> <domainY> <transformationNum>
<domainX> <domainY> <transformationNum>
...
... (one line for each Range Block)
...
<domainX> <domainY> <transformationNum>
```

##Notes
We are choosing to save our files as ASCII. We could save them as binary instead, but since this would be the same for both parallelized, and sequential implementations, we decided it was not important to worry about for now.
Additionally, we could save the codebook elements (range blocks) in a look up table and have each line simply reference an element in the table (rather than redundantly repeat information). Again, this change will not differentiate the parallel and sequential implementations so we decided not to implement it for now.
