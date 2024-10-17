Variable
========

.. meta::
   :description: Reference page for GAMSPy Variable (gamspy.Variable) and GAMSPy VariableType (gamspy.VariableType)
   :keywords: Reference, Variable, VariableType, GAMSPy, gamspy, GAMS, gams, mathematical modeling, sparsity, performance

.. autoclass:: gamspy.Variable
   :members: synchronize,computeInfeasibilities,container,countEps,countNA,countNegInf,countPosInf,countUndef,default_records,description,dimension,domain,domain_forwarding,domain_labels,domain_names,domain_type,dropDefaults,dropEps,dropMissing,dropNA,dropUndef,equals,findEps,findNA,findNegInf,findPosInf,findSpecialValues,findUndef,fx,gamsRepr,generateRecords,getVariableListing,getDeclaration,getAssignment,getMaxAbsValue,getMaxValue,getMeanValue,getMinValue,getSparsity,isValid,is_scalar,l,lo,m,modified,name,number_records,pivot,prior,records,scale,setRecords,shape,stage,summary,toDense,toDict,toList,toSparseCoo,toValue,type,up,where,whereMax,whereMaxAbs,whereMin
   :undoc-members:
   :show-inheritance:

.. autoclass:: gamspy.VariableType
   :members: BINARY,INTEGER,POSITIVE,NEGATIVE,FREE,SOS1,SOS2,SEMICONT,SEMIINT
   :undoc-members:
   :show-inheritance:
