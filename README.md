# HyperKGQA_qualitative
Modules for qualitative analysis of HyperKGQA

## For MetaQA:
Run relation_performance_MetaQA.py specifying the hops and whether you need to preprocess and get the curvature beforehand.
For example:
```
python relation_performance_MetaQA.py --hops 3 --preprocessing True --get_curvature True
```


## For WebQuestionsSP:
Run relation_performance_fbwq.py specifying whether you need to preprocess and get the curvature beforehand. Then run fbwq_analysis.py to group the different inference paths in curvature ranges.
For example:
```
python relation_performance_fbwq.py --preprocessing True --get_curvature True
python fbwq_analysis.py
```
