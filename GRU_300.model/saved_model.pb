Á
ùÊ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68í¢
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:2*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	.*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	.*
dtype0

gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*.
shared_namegru/gru_cell/recurrent_kernel

1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes
:	2*
dtype0

gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namegru/gru_cell/bias
x
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes
:	*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:2*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	.*+
shared_nameAdam/gru/gru_cell/kernel/m

.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/m*
_output_shapes
:	.*
dtype0
¥
$Adam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/m

8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/m*
_output_shapes
:	2*
dtype0

Adam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/gru/gru_cell/bias/m

,Adam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/m*
_output_shapes
:	*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:2*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	.*+
shared_nameAdam/gru/gru_cell/kernel/v

.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/v*
_output_shapes
:	.*
dtype0
¥
$Adam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/v

8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/v*
_output_shapes
:	2*
dtype0

Adam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/gru/gru_cell/bias/v

,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/v*
_output_shapes
:	*
dtype0

NoOpNoOp
à'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*'
value'B' B'

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
Á
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

iter

beta_1

beta_2
	 decay
!learning_ratemGmH"mI#mJ$mKvLvM"vN#vO$vP*
'
"0
#1
$2
3
4*
'
"0
#1
$2
3
4*
* 
°
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
* 
* 
* 

*serving_default* 
Ó

"kernel
#recurrent_kernel
$bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses*
* 

"0
#1
$2*

"0
#1
$2*
* 


2states
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEgru/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEgru/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

=0*
* 
* 
* 

"0
#1
$2*

"0
#1
$2*
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ctotal
	Dcount
E	variables
F	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

E	variables*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/gru/gru_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/gru/gru_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_gru_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ.

StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_inputgru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kerneldense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_8864
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
£	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp,Adam/gru/gru_cell/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_9734
æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/gru/gru_cell/kernel/m$Adam/gru/gru_cell/recurrent_kernel/mAdam/gru/gru_cell/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/gru/gru_cell/kernel/v$Adam/gru/gru_cell/recurrent_kernel/vAdam/gru/gru_cell/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_9810¼
 @
ã
gru_while_body_8593$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	F
3gru_while_gru_cell_matmul_readvariableop_resource_0:	.H
5gru_while_gru_cell_matmul_1_readvariableop_resource_0:	2
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	D
1gru_while_gru_cell_matmul_readvariableop_resource:	.F
3gru_while_gru_cell_matmul_1_readvariableop_resource:	2¢(gru/while/gru_cell/MatMul/ReadVariableOp¢*gru/while/gru_cell/MatMul_1/ReadVariableOp¢!gru/while/gru_cell/ReadVariableOp
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   º
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0¾
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÜ
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split¡
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0¥
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿo
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ñ
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒQ
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: ¥
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0^gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ì
gru/while/NoOpNoOp)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"¸
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
;
ë
while_body_8088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	.D
1while_gru_cell_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	.B
/while_gru_cell_matmul_1_readvariableop_resource:	2¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÐ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Á
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒu
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
ï
³
"__inference_gru_layer_call_fn_8908

inputs
unknown:	
	unknown_0:	.
	unknown_1:	2
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_8391o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ð
 
while_cond_9430
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_9430___redundant_placeholder02
.while_while_cond_9430___redundant_placeholder12
.while_while_cond_9430___redundant_placeholder22
.while_while_cond_9430___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
©K
ñ
=__inference_gru_layer_call_and_return_conditional_losses_8391

inputs3
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	.<
)gru_cell_matmul_1_readvariableop_resource:	2
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8302*
condR
while_cond_8301*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ú
ó
)__inference_sequential_layer_call_fn_8461
	gru_input
unknown:	
	unknown_0:	.
	unknown_1:	2
	unknown_2:2
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_8433o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
#
_user_specified_name	gru_input
g

__inference__wrapped_model_7680
	gru_inputB
/sequential_gru_gru_cell_readvariableop_resource:	I
6sequential_gru_gru_cell_matmul_readvariableop_resource:	.K
8sequential_gru_gru_cell_matmul_1_readvariableop_resource:	2A
/sequential_dense_matmul_readvariableop_resource:2>
0sequential_dense_biasadd_readvariableop_resource:
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢-sequential/gru/gru_cell/MatMul/ReadVariableOp¢/sequential/gru/gru_cell/MatMul_1/ReadVariableOp¢&sequential/gru/gru_cell/ReadVariableOp¢sequential/gru/whileM
sequential/gru/ShapeShape	gru_input*
T0*
_output_shapes
:l
"sequential/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$sequential/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$sequential/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
sequential/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
sequential/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
sequential/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential/gru/transpose	Transpose	gru_input&sequential/gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.b
sequential/gru/Shape_1Shapesequential/gru/transpose:y:0*
T0*
_output_shapes
:n
$sequential/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&sequential/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&sequential/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
sequential/gru/strided_slice_1StridedSlicesequential/gru/Shape_1:output:0-sequential/gru/strided_slice_1/stack:output:0/sequential/gru/strided_slice_1/stack_1:output:0/sequential/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*sequential/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿá
sequential/gru/TensorArrayV2TensorListReserve3sequential/gru/TensorArrayV2/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   
6sequential/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/gru/transpose:y:0Msequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒn
$sequential/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&sequential/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&sequential/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:´
sequential/gru/strided_slice_2StridedSlicesequential/gru/transpose:y:0-sequential/gru/strided_slice_2/stack:output:0/sequential/gru/strided_slice_2/stack_1:output:0/sequential/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_mask
&sequential/gru/gru_cell/ReadVariableOpReadVariableOp/sequential_gru_gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0
sequential/gru/gru_cell/unstackUnpack.sequential/gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num¥
-sequential/gru/gru_cell/MatMul/ReadVariableOpReadVariableOp6sequential_gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0»
sequential/gru/gru_cell/MatMulMatMul'sequential/gru/strided_slice_2:output:05sequential/gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
sequential/gru/gru_cell/BiasAddBiasAdd(sequential/gru/gru_cell/MatMul:product:0(sequential/gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'sequential/gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
sequential/gru/gru_cell/splitSplit0sequential/gru/gru_cell/split/split_dim:output:0(sequential/gru/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split©
/sequential/gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp8sequential_gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0µ
 sequential/gru/gru_cell/MatMul_1MatMulsequential/gru/zeros:output:07sequential/gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
!sequential/gru/gru_cell/BiasAdd_1BiasAdd*sequential/gru/gru_cell/MatMul_1:product:0(sequential/gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
sequential/gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿt
)sequential/gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
sequential/gru/gru_cell/split_1SplitV*sequential/gru/gru_cell/BiasAdd_1:output:0&sequential/gru/gru_cell/Const:output:02sequential/gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split¨
sequential/gru/gru_cell/addAddV2&sequential/gru/gru_cell/split:output:0(sequential/gru/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
sequential/gru/gru_cell/SigmoidSigmoidsequential/gru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ª
sequential/gru/gru_cell/add_1AddV2&sequential/gru/gru_cell/split:output:1(sequential/gru/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
!sequential/gru/gru_cell/Sigmoid_1Sigmoid!sequential/gru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¥
sequential/gru/gru_cell/mulMul%sequential/gru/gru_cell/Sigmoid_1:y:0(sequential/gru/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¡
sequential/gru/gru_cell/add_2AddV2&sequential/gru/gru_cell/split:output:2sequential/gru/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2y
sequential/gru/gru_cell/ReluRelu!sequential/gru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/gru/gru_cell/mul_1Mul#sequential/gru/gru_cell/Sigmoid:y:0sequential/gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
sequential/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
sequential/gru/gru_cell/subSub&sequential/gru/gru_cell/sub/x:output:0#sequential/gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2£
sequential/gru/gru_cell/mul_2Mulsequential/gru/gru_cell/sub:z:0*sequential/gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/gru/gru_cell/add_3AddV2!sequential/gru/gru_cell/mul_1:z:0!sequential/gru/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
,sequential/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   å
sequential/gru/TensorArrayV2_1TensorListReserve5sequential/gru/TensorArrayV2_1/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
sequential/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'sequential/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!sequential/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
sequential/gru/whileWhile*sequential/gru/while/loop_counter:output:00sequential/gru/while/maximum_iterations:output:0sequential/gru/time:output:0'sequential/gru/TensorArrayV2_1:handle:0sequential/gru/zeros:output:0'sequential/gru/strided_slice_1:output:0Fsequential/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/sequential_gru_gru_cell_readvariableop_resource6sequential_gru_gru_cell_matmul_readvariableop_resource8sequential_gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
sequential_gru_while_body_7585**
cond"R 
sequential_gru_while_cond_7584*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ï
1sequential/gru/TensorArrayV2Stack/TensorListStackTensorListStacksequential/gru/while:output:3Hsequential/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0w
$sequential/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&sequential/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&sequential/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
sequential/gru/strided_slice_3StridedSlice:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0-sequential/gru/strided_slice_3/stack:output:0/sequential/gru/strided_slice_3/stack_1:output:0/sequential/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskt
sequential/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
sequential/gru/transpose_1	Transpose:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0(sequential/gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
sequential/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0¬
sequential/dense/MatMulMatMul'sequential/gru/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
IdentityIdentity!sequential/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp.^sequential/gru/gru_cell/MatMul/ReadVariableOp0^sequential/gru/gru_cell/MatMul_1/ReadVariableOp'^sequential/gru/gru_cell/ReadVariableOp^sequential/gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2^
-sequential/gru/gru_cell/MatMul/ReadVariableOp-sequential/gru/gru_cell/MatMul/ReadVariableOp2b
/sequential/gru/gru_cell/MatMul_1/ReadVariableOp/sequential/gru/gru_cell/MatMul_1/ReadVariableOp2P
&sequential/gru/gru_cell/ReadVariableOp&sequential/gru/gru_cell/ReadVariableOp2,
sequential/gru/whilesequential/gru/while:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
#
_user_specified_name	gru_input
Ð
 
while_cond_8301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8301___redundant_placeholder02
.while_while_cond_8301___redundant_placeholder12
.while_while_cond_8301___redundant_placeholder22
.while_while_cond_8301___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
¦

while_body_7763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_gru_cell_7785_0:	(
while_gru_cell_7787_0:	.(
while_gru_cell_7789_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_gru_cell_7785:	&
while_gru_cell_7787:	.&
while_gru_cell_7789:	2¢&while/gru_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0í
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_7785_0while_gru_cell_7787_0while_gru_cell_7789_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_7750Ø
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ",
while_gru_cell_7785while_gru_cell_7785_0",
while_gru_cell_7787while_gru_cell_7787_0",
while_gru_cell_7789while_gru_cell_7789_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ð
 
while_cond_8971
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8971___redundant_placeholder02
.while_while_cond_8971___redundant_placeholder12
.while_while_cond_8971___redundant_placeholder22
.while_while_cond_8971___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
¦

while_body_7945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0(
while_gru_cell_7967_0:	(
while_gru_cell_7969_0:	.(
while_gru_cell_7971_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor&
while_gru_cell_7967:	&
while_gru_cell_7969:	.&
while_gru_cell_7971:	2¢&while/gru_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0í
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_7967_0while_gru_cell_7969_0while_gru_cell_7971_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_7893Ø
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ",
while_gru_cell_7967while_gru_cell_7967_0",
while_gru_cell_7969while_gru_cell_7969_0",
while_gru_cell_7971while_gru_cell_7971_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
æ

½
sequential_gru_while_cond_7584:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_2<
8sequential_gru_while_less_sequential_gru_strided_slice_1P
Lsequential_gru_while_sequential_gru_while_cond_7584___redundant_placeholder0P
Lsequential_gru_while_sequential_gru_while_cond_7584___redundant_placeholder1P
Lsequential_gru_while_sequential_gru_while_cond_7584___redundant_placeholder2P
Lsequential_gru_while_sequential_gru_while_cond_7584___redundant_placeholder3!
sequential_gru_while_identity

sequential/gru/while/LessLess sequential_gru_while_placeholder8sequential_gru_while_less_sequential_gru_strided_slice_1*
T0*
_output_shapes
: i
sequential/gru/while/IdentityIdentitysequential/gru/while/Less:z:0*
T0
*
_output_shapes
: "G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
;
ë
while_body_8972
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	.D
1while_gru_cell_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	.B
/while_gru_cell_matmul_1_readvariableop_resource:	2¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÐ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Á
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒu
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 

Ö
B__inference_gru_cell_layer_call_and_return_conditional_losses_9645

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	.3
 matmul_1_readvariableop_resource:	2
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	.*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ.:ÿÿÿÿÿÿÿÿÿ2: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0

Ô
B__inference_gru_cell_layer_call_and_return_conditional_losses_7893

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	.3
 matmul_1_readvariableop_resource:	2
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	.*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ.:ÿÿÿÿÿÿÿÿÿ2: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_namestates
N
­
sequential_gru_while_body_7585:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_29
5sequential_gru_while_sequential_gru_strided_slice_1_0u
qsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0J
7sequential_gru_while_gru_cell_readvariableop_resource_0:	Q
>sequential_gru_while_gru_cell_matmul_readvariableop_resource_0:	.S
@sequential_gru_while_gru_cell_matmul_1_readvariableop_resource_0:	2!
sequential_gru_while_identity#
sequential_gru_while_identity_1#
sequential_gru_while_identity_2#
sequential_gru_while_identity_3#
sequential_gru_while_identity_47
3sequential_gru_while_sequential_gru_strided_slice_1s
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensorH
5sequential_gru_while_gru_cell_readvariableop_resource:	O
<sequential_gru_while_gru_cell_matmul_readvariableop_resource:	.Q
>sequential_gru_while_gru_cell_matmul_1_readvariableop_resource:	2¢3sequential/gru/while/gru_cell/MatMul/ReadVariableOp¢5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp¢,sequential/gru/while/gru_cell/ReadVariableOp
Fsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ñ
8sequential/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0 sequential_gru_while_placeholderOsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0¥
,sequential/gru/while/gru_cell/ReadVariableOpReadVariableOp7sequential_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
%sequential/gru/while/gru_cell/unstackUnpack4sequential/gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num³
3sequential/gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp>sequential_gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0ß
$sequential/gru/while/gru_cell/MatMulMatMul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential/gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
%sequential/gru/while/gru_cell/BiasAddBiasAdd.sequential/gru/while/gru_cell/MatMul:product:0.sequential/gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
-sequential/gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿý
#sequential/gru/while/gru_cell/splitSplit6sequential/gru/while/gru_cell/split/split_dim:output:0.sequential/gru/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split·
5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp@sequential_gru_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0Æ
&sequential/gru/while/gru_cell/MatMul_1MatMul"sequential_gru_while_placeholder_2=sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
'sequential/gru/while/gru_cell/BiasAdd_1BiasAdd0sequential/gru/while/gru_cell/MatMul_1:product:0.sequential/gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
#sequential/gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿz
/sequential/gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
%sequential/gru/while/gru_cell/split_1SplitV0sequential/gru/while/gru_cell/BiasAdd_1:output:0,sequential/gru/while/gru_cell/Const:output:08sequential/gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitº
!sequential/gru/while/gru_cell/addAddV2,sequential/gru/while/gru_cell/split:output:0.sequential/gru/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
%sequential/gru/while/gru_cell/SigmoidSigmoid%sequential/gru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼
#sequential/gru/while/gru_cell/add_1AddV2,sequential/gru/while/gru_cell/split:output:1.sequential/gru/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
'sequential/gru/while/gru_cell/Sigmoid_1Sigmoid'sequential/gru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2·
!sequential/gru/while/gru_cell/mulMul+sequential/gru/while/gru_cell/Sigmoid_1:y:0.sequential/gru/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2³
#sequential/gru/while/gru_cell/add_2AddV2,sequential/gru/while/gru_cell/split:output:2%sequential/gru/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"sequential/gru/while/gru_cell/ReluRelu'sequential/gru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2«
#sequential/gru/while/gru_cell/mul_1Mul)sequential/gru/while/gru_cell/Sigmoid:y:0"sequential_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2h
#sequential/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
!sequential/gru/while/gru_cell/subSub,sequential/gru/while/gru_cell/sub/x:output:0)sequential/gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2µ
#sequential/gru/while/gru_cell/mul_2Mul%sequential/gru/while/gru_cell/sub:z:00sequential/gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2°
#sequential/gru/while/gru_cell/add_3AddV2'sequential/gru/while/gru_cell/mul_1:z:0'sequential/gru/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ý
9sequential/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"sequential_gru_while_placeholder_1 sequential_gru_while_placeholder'sequential/gru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ\
sequential/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential/gru/while/addAddV2 sequential_gru_while_placeholder#sequential/gru/while/add/y:output:0*
T0*
_output_shapes
: ^
sequential/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
sequential/gru/while/add_1AddV26sequential_gru_while_sequential_gru_while_loop_counter%sequential/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 
sequential/gru/while/IdentityIdentitysequential/gru/while/add_1:z:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: ¦
sequential/gru/while/Identity_1Identity<sequential_gru_while_sequential_gru_while_maximum_iterations^sequential/gru/while/NoOp*
T0*
_output_shapes
: 
sequential/gru/while/Identity_2Identitysequential/gru/while/add:z:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: Æ
sequential/gru/while/Identity_3IdentityIsequential/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: :éèÒ¢
sequential/gru/while/Identity_4Identity'sequential/gru/while/gru_cell/add_3:z:0^sequential/gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ø
sequential/gru/while/NoOpNoOp4^sequential/gru/while/gru_cell/MatMul/ReadVariableOp6^sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp-^sequential/gru/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
>sequential_gru_while_gru_cell_matmul_1_readvariableop_resource@sequential_gru_while_gru_cell_matmul_1_readvariableop_resource_0"~
<sequential_gru_while_gru_cell_matmul_readvariableop_resource>sequential_gru_while_gru_cell_matmul_readvariableop_resource_0"p
5sequential_gru_while_gru_cell_readvariableop_resource7sequential_gru_while_gru_cell_readvariableop_resource_0"G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0"K
sequential_gru_while_identity_1(sequential/gru/while/Identity_1:output:0"K
sequential_gru_while_identity_2(sequential/gru/while/Identity_2:output:0"K
sequential_gru_while_identity_3(sequential/gru/while/Identity_3:output:0"K
sequential_gru_while_identity_4(sequential/gru/while/Identity_4:output:0"l
3sequential_gru_while_sequential_gru_strided_slice_15sequential_gru_while_sequential_gru_strided_slice_1_0"ä
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensorqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2j
3sequential/gru/while/gru_cell/MatMul/ReadVariableOp3sequential/gru/while/gru_cell/MatMul/ReadVariableOp2n
5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp2\
,sequential/gru/while/gru_cell/ReadVariableOp,sequential/gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
©K
ñ
=__inference_gru_layer_call_and_return_conditional_losses_8177

inputs3
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	.<
)gru_cell_matmul_1_readvariableop_resource:	2
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8088*
condR
while_cond_8087*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs

µ
"__inference_gru_layer_call_fn_8886
inputs_0
unknown:	
	unknown_0:	.
	unknown_1:	2
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_8009o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.
"
_user_specified_name
inputs/0
å
°
D__inference_sequential_layer_call_and_return_conditional_losses_8433

inputs
gru_8420:	
gru_8422:	.
gru_8424:	2

dense_8427:2

dense_8429:
identity¢dense/StatefulPartitionedCall¢gru/StatefulPartitionedCallâ
gru/StatefulPartitionedCallStatefulPartitionedCallinputsgru_8420gru_8422gru_8424*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_8391ü
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0
dense_8427
dense_8429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8195u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
ï
³
"__inference_gru_layer_call_fn_8897

inputs
unknown:	
	unknown_0:	.
	unknown_1:	2
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_8177o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ú
ó
)__inference_sequential_layer_call_fn_8215
	gru_input
unknown:	
	unknown_0:	.
	unknown_1:	2
	unknown_2:2
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_8202o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
#
_user_specified_name	gru_input
Ð
 
while_cond_8087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8087___redundant_placeholder02
.while_while_cond_8087___redundant_placeholder12
.while_while_cond_8087___redundant_placeholder22
.while_while_cond_8087___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:

Ô
B__inference_gru_cell_layer_call_and_return_conditional_losses_7750

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	.3
 matmul_1_readvariableop_resource:	2
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	.*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ.:ÿÿÿÿÿÿÿÿÿ2: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_namestates
Ð
 
while_cond_7944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_7944___redundant_placeholder02
.while_while_cond_7944___redundant_placeholder12
.while_while_cond_7944___redundant_placeholder22
.while_while_cond_7944___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
Ã3
í
=__inference_gru_layer_call_and_return_conditional_losses_7827

inputs 
gru_cell_7751:	 
gru_cell_7753:	. 
gru_cell_7755:	2
identity¢ gru_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_mask²
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_7751gru_cell_7753gru_cell_7755*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_7750n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : è
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_7751gru_cell_7753gru_cell_7755*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_7763*
condR
while_cond_7762*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs

µ
"__inference_gru_layer_call_fn_8875
inputs_0
unknown:	
	unknown_0:	.
	unknown_1:	2
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_7827o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.
"
_user_specified_name
inputs/0
ÞK
ó
=__inference_gru_layer_call_and_return_conditional_losses_9214
inputs_03
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	.<
)gru_cell_matmul_1_readvariableop_resource:	2
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_9125*
condR
while_cond_9124*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.
"
_user_specified_name
inputs/0
î
³
D__inference_sequential_layer_call_and_return_conditional_losses_8477
	gru_input
gru_8464:	
gru_8466:	.
gru_8468:	2

dense_8471:2

dense_8473:
identity¢dense/StatefulPartitionedCall¢gru/StatefulPartitionedCallå
gru/StatefulPartitionedCallStatefulPartitionedCall	gru_inputgru_8464gru_8466gru_8468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_8177ü
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0
dense_8471
dense_8473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8195u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
#
_user_specified_name	gru_input
;
ë
while_body_8302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	.D
1while_gru_cell_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	.B
/while_gru_cell_matmul_1_readvariableop_resource:	2¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÐ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Á
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒu
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
¸

$__inference_dense_layer_call_fn_9529

inputs
unknown:2
	unknown_0:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8195o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ÙY
Ø
 __inference__traced_restore_9810
file_prefix/
assignvariableop_dense_kernel:2+
assignvariableop_1_dense_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: 9
&assignvariableop_7_gru_gru_cell_kernel:	.C
0assignvariableop_8_gru_gru_cell_recurrent_kernel:	27
$assignvariableop_9_gru_gru_cell_bias:	#
assignvariableop_10_total: #
assignvariableop_11_count: 9
'assignvariableop_12_adam_dense_kernel_m:23
%assignvariableop_13_adam_dense_bias_m:A
.assignvariableop_14_adam_gru_gru_cell_kernel_m:	.K
8assignvariableop_15_adam_gru_gru_cell_recurrent_kernel_m:	2?
,assignvariableop_16_adam_gru_gru_cell_bias_m:	9
'assignvariableop_17_adam_dense_kernel_v:23
%assignvariableop_18_adam_dense_bias_v:A
.assignvariableop_19_adam_gru_gru_cell_kernel_v:	.K
8assignvariableop_20_adam_gru_gru_cell_recurrent_kernel_v:	2?
,assignvariableop_21_adam_gru_gru_cell_bias_v:	
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¨
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î

valueÄ
BÁ
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp&assignvariableop_7_gru_gru_cell_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp0assignvariableop_8_gru_gru_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_gru_gru_cell_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_adam_dense_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_adam_dense_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp.assignvariableop_14_adam_gru_gru_cell_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_15AssignVariableOp8assignvariableop_15_adam_gru_gru_cell_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp,assignvariableop_16_adam_gru_gru_cell_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_gru_gru_cell_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_gru_gru_cell_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_gru_gru_cell_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
î
³
D__inference_sequential_layer_call_and_return_conditional_losses_8493
	gru_input
gru_8480:	
gru_8482:	.
gru_8484:	2

dense_8487:2

dense_8489:
identity¢dense/StatefulPartitionedCall¢gru/StatefulPartitionedCallå
gru/StatefulPartitionedCallStatefulPartitionedCall	gru_inputgru_8480gru_8482gru_8484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_8391ü
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0
dense_8487
dense_8489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8195u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
#
_user_specified_name	gru_input
©K
ñ
=__inference_gru_layer_call_and_return_conditional_losses_9520

inputs3
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	.<
)gru_cell_matmul_1_readvariableop_resource:	2
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_9431*
condR
while_cond_9430*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
 @
ã
gru_while_body_8752$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0?
,gru_while_gru_cell_readvariableop_resource_0:	F
3gru_while_gru_cell_matmul_readvariableop_resource_0:	.H
5gru_while_gru_cell_matmul_1_readvariableop_resource_0:	2
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor=
*gru_while_gru_cell_readvariableop_resource:	D
1gru_while_gru_cell_matmul_readvariableop_resource:	.F
3gru_while_gru_cell_matmul_1_readvariableop_resource:	2¢(gru/while/gru_cell/MatMul/ReadVariableOp¢*gru/while/gru_cell/MatMul_1/ReadVariableOp¢!gru/while/gru_cell/ReadVariableOp
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   º
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0¾
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÜ
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split¡
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0¥
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
gru/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿo
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0!gru/while/gru_cell/Const:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ñ
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒQ
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: ¥
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0^gru/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ì
gru/while/NoOpNoOp)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"¸
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ð
 
while_cond_7762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_7762___redundant_placeholder02
.while_while_cond_7762___redundant_placeholder12
.while_while_cond_7762___redundant_placeholder22
.while_while_cond_7762___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
Ð
 
while_cond_9124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_9124___redundant_placeholder02
.while_while_cond_9124___redundant_placeholder12
.while_while_cond_9124___redundant_placeholder22
.while_while_cond_9124___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
ª

Ö
'__inference_gru_cell_layer_call_fn_9553

inputs
states_0
unknown:	
	unknown_0:	.
	unknown_1:	2
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_7750o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ.:ÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0
;
ë
while_body_9278
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	.D
1while_gru_cell_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	.B
/while_gru_cell_matmul_1_readvariableop_resource:	2¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÐ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Á
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒu
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ñ
ð
)__inference_sequential_layer_call_fn_8529

inputs
unknown:	
	unknown_0:	.
	unknown_1:	2
	unknown_2:2
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_8433o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
ÂW
¾
D__inference_sequential_layer_call_and_return_conditional_losses_8688

inputs7
$gru_gru_cell_readvariableop_resource:	>
+gru_gru_cell_matmul_readvariableop_resource:	.@
-gru_gru_cell_matmul_1_readvariableop_resource:	26
$dense_matmul_readvariableop_resource:23
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢"gru/gru_cell/MatMul/ReadVariableOp¢$gru/gru_cell/MatMul_1/ReadVariableOp¢gru/gru_cell/ReadVariableOp¢	gru/while?
	gru/ShapeShapeinputs*
T0*
_output_shapes
:a
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.L
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:c
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ì
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒc
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ý
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_mask
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0{
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÊ
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru/gru_cell/MatMul_1MatMulgru/zeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿi
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿú
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2y
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2W
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ä
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒJ
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿX
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : å
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
gru_while_body_8593*
condR
gru_while_cond_8592*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Î
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿe
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¢
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense/MatMulMatMulgru/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
ÞK
ó
=__inference_gru_layer_call_and_return_conditional_losses_9061
inputs_03
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	.<
)gru_cell_matmul_1_readvariableop_resource:	2
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_8972*
condR
while_cond_8971*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.
"
_user_specified_name
inputs/0
Â	
ð
?__inference_dense_layer_call_and_return_conditional_losses_9539

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
;
ë
while_body_9125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	.D
1while_gru_cell_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	.B
/while_gru_cell_matmul_1_readvariableop_resource:	2¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÐ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Á
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒu
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
ÂW
¾
D__inference_sequential_layer_call_and_return_conditional_losses_8847

inputs7
$gru_gru_cell_readvariableop_resource:	>
+gru_gru_cell_matmul_readvariableop_resource:	.@
-gru_gru_cell_matmul_1_readvariableop_resource:	26
$dense_matmul_readvariableop_resource:23
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢"gru/gru_cell/MatMul/ReadVariableOp¢$gru/gru_cell/MatMul_1/ReadVariableOp¢gru/gru_cell/ReadVariableOp¢	gru/while?
	gru/ShapeShapeinputs*
T0*
_output_shapes
:a
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.L
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:c
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ì
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒc
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ý
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_mask
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0{
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÊ
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru/gru_cell/MatMul_1MatMulgru/zeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
gru/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿi
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿú
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2y
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2W
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ä
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒJ
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿX
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : å
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
gru_while_body_8752*
condR
gru_while_cond_8751*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Î
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿe
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¢
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense/MatMulMatMulgru/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿù
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
¼
ì
gru_while_cond_8592$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1:
6gru_while_gru_while_cond_8592___redundant_placeholder0:
6gru_while_gru_while_cond_8592___redundant_placeholder1:
6gru_while_gru_while_cond_8592___redundant_placeholder2:
6gru_while_gru_while_cond_8592___redundant_placeholder3
gru_while_identity
r
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:

Ö
B__inference_gru_cell_layer_call_and_return_conditional_losses_9606

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	.3
 matmul_1_readvariableop_resource:	2
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	.*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ.:ÿÿÿÿÿÿÿÿÿ2: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0
4
Ð	
__inference__traced_save_9734
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_m_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_v_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¥
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î

valueÄ
BÁ
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B Ô	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop5savev2_adam_gru_gru_cell_kernel_m_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop3savev2_adam_gru_gru_cell_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop5savev2_adam_gru_gru_cell_kernel_v_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop3savev2_adam_gru_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*º
_input_shapes¨
¥: :2:: : : : : :	.:	2:	: : :2::	.:	2:	:2::	.:	2:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	.:%	!

_output_shapes
:	2:%
!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	.:%!

_output_shapes
:	2:%!

_output_shapes
:	:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	.:%!

_output_shapes
:	2:%!

_output_shapes
:	:

_output_shapes
: 
Ñ
ð
)__inference_sequential_layer_call_fn_8514

inputs
unknown:	
	unknown_0:	.
	unknown_1:	2
	unknown_2:2
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_8202o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
®
ì
"__inference_signature_wrapper_8864
	gru_input
unknown:	
	unknown_0:	.
	unknown_1:	2
	unknown_2:2
	unknown_3:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_7680o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
#
_user_specified_name	gru_input
ª

Ö
'__inference_gru_cell_layer_call_fn_9567

inputs
states_0
unknown:	
	unknown_0:	.
	unknown_1:	2
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_7893o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ.:ÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0
;
ë
while_body_9431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	B
/while_gru_cell_matmul_readvariableop_resource_0:	.D
1while_gru_cell_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	@
-while_gru_cell_matmul_readvariableop_resource:	.B
/while_gru_cell_matmul_1_readvariableop_resource:	2¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
element_dtype0
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	.*
dtype0²
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÐ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿk
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2g
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Á
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒu
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¼

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
©K
ñ
=__inference_gru_layer_call_and_return_conditional_losses_9367

inputs3
 gru_cell_readvariableop_resource:	:
'gru_cell_matmul_readvariableop_resource:	.<
)gru_cell_matmul_1_readvariableop_resource:	2
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	.*
dtype0
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿe
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿê
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ±
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_9278*
condR
while_cond_9277*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ.: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Â	
ð
?__inference_dense_layer_call_and_return_conditional_losses_8195

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ã3
í
=__inference_gru_layer_call_and_return_conditional_losses_8009

inputs 
gru_cell_7933:	 
gru_cell_7935:	. 
gru_cell_7937:	2
identity¢ gru_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ.   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.*
shrink_axis_mask²
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_7933gru_cell_7935gru_cell_7937*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_7893n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : è
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_7933gru_cell_7935gru_cell_7937*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_7945*
condR
while_cond_7944*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
å
°
D__inference_sequential_layer_call_and_return_conditional_losses_8202

inputs
gru_8178:	
gru_8180:	.
gru_8182:	2

dense_8196:2

dense_8198:
identity¢dense/StatefulPartitionedCall¢gru/StatefulPartitionedCallâ
gru/StatefulPartitionedCallStatefulPartitionedCallinputsgru_8178gru_8180gru_8182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_8177ü
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0
dense_8196
dense_8198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8195u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ.: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ.
 
_user_specified_nameinputs
Ð
 
while_cond_9277
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_9277___redundant_placeholder02
.while_while_cond_9277___redundant_placeholder12
.while_while_cond_9277___redundant_placeholder22
.while_while_cond_9277___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
¼
ì
gru_while_cond_8751$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1:
6gru_while_gru_while_cond_8751___redundant_placeholder0:
6gru_while_gru_while_cond_8751___redundant_placeholder1:
6gru_while_gru_while_cond_8751___redundant_placeholder2:
6gru_while_gru_while_cond_8751___redundant_placeholder3
gru_while_identity
r
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*°
serving_default
C
	gru_input6
serving_default_gru_input:0ÿÿÿÿÿÿÿÿÿ.9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ï^
´
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_sequential
Ú
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
­
iter

beta_1

beta_2
	 decay
!learning_ratemGmH"mI#mJ$mKvLvM"vN#vO$vP"
	optimizer
C
"0
#1
$2
3
4"
trackable_list_wrapper
C
"0
#1
$2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_sequential_layer_call_fn_8215
)__inference_sequential_layer_call_fn_8514
)__inference_sequential_layer_call_fn_8529
)__inference_sequential_layer_call_fn_8461À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_sequential_layer_call_and_return_conditional_losses_8688
D__inference_sequential_layer_call_and_return_conditional_losses_8847
D__inference_sequential_layer_call_and_return_conditional_losses_8477
D__inference_sequential_layer_call_and_return_conditional_losses_8493À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
__inference__wrapped_model_7680	gru_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
*serving_default"
signature_map
è

"kernel
#recurrent_kernel
$bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

2states
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
ë2è
"__inference_gru_layer_call_fn_8875
"__inference_gru_layer_call_fn_8886
"__inference_gru_layer_call_fn_8897
"__inference_gru_layer_call_fn_8908Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×2Ô
=__inference_gru_layer_call_and_return_conditional_losses_9061
=__inference_gru_layer_call_and_return_conditional_losses_9214
=__inference_gru_layer_call_and_return_conditional_losses_9367
=__inference_gru_layer_call_and_return_conditional_losses_9520Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:22dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_dense_layer_call_fn_9529¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_9539¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
&:$	.2gru/gru_cell/kernel
0:.	22gru/gru_cell/recurrent_kernel
$:"	2gru/gru_cell/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
"__inference_signature_wrapper_8864	gru_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
"0
#1
$2"
trackable_list_wrapper
5
"0
#1
$2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
'__inference_gru_cell_layer_call_fn_9553
'__inference_gru_cell_layer_call_fn_9567¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
B__inference_gru_cell_layer_call_and_return_conditional_losses_9606
B__inference_gru_cell_layer_call_and_return_conditional_losses_9645¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ctotal
	Dcount
E	variables
F	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
C0
D1"
trackable_list_wrapper
-
E	variables"
_generic_user_object
#:!22Adam/dense/kernel/m
:2Adam/dense/bias/m
+:)	.2Adam/gru/gru_cell/kernel/m
5:3	22$Adam/gru/gru_cell/recurrent_kernel/m
):'	2Adam/gru/gru_cell/bias/m
#:!22Adam/dense/kernel/v
:2Adam/dense/bias/v
+:)	.2Adam/gru/gru_cell/kernel/v
5:3	22$Adam/gru/gru_cell/recurrent_kernel/v
):'	2Adam/gru/gru_cell/bias/v
__inference__wrapped_model_7680n$"#6¢3
,¢)
'$
	gru_inputÿÿÿÿÿÿÿÿÿ.
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ
?__inference_dense_layer_call_and_return_conditional_losses_9539\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
$__inference_dense_layer_call_fn_9529O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿþ
B__inference_gru_cell_layer_call_and_return_conditional_losses_9606·$"#\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ.
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ2
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ2
$!

0/1/0ÿÿÿÿÿÿÿÿÿ2
 þ
B__inference_gru_cell_layer_call_and_return_conditional_losses_9645·$"#\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ.
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ2
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ2
$!

0/1/0ÿÿÿÿÿÿÿÿÿ2
 Õ
'__inference_gru_cell_layer_call_fn_9553©$"#\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ.
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ2
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ2
"

1/0ÿÿÿÿÿÿÿÿÿ2Õ
'__inference_gru_cell_layer_call_fn_9567©$"#\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ.
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ2
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ2
"

1/0ÿÿÿÿÿÿÿÿÿ2¾
=__inference_gru_layer_call_and_return_conditional_losses_9061}$"#O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ¾
=__inference_gru_layer_call_and_return_conditional_losses_9214}$"#O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ®
=__inference_gru_layer_call_and_return_conditional_losses_9367m$"#?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ.

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ®
=__inference_gru_layer_call_and_return_conditional_losses_9520m$"#?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ.

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 
"__inference_gru_layer_call_fn_8875p$"#O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2
"__inference_gru_layer_call_fn_8886p$"#O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ.

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ2
"__inference_gru_layer_call_fn_8897`$"#?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ.

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2
"__inference_gru_layer_call_fn_8908`$"#?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ.

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ2¶
D__inference_sequential_layer_call_and_return_conditional_losses_8477n$"#>¢;
4¢1
'$
	gru_inputÿÿÿÿÿÿÿÿÿ.
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
D__inference_sequential_layer_call_and_return_conditional_losses_8493n$"#>¢;
4¢1
'$
	gru_inputÿÿÿÿÿÿÿÿÿ.
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ³
D__inference_sequential_layer_call_and_return_conditional_losses_8688k$"#;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ.
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ³
D__inference_sequential_layer_call_and_return_conditional_losses_8847k$"#;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ.
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_sequential_layer_call_fn_8215a$"#>¢;
4¢1
'$
	gru_inputÿÿÿÿÿÿÿÿÿ.
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_8461a$"#>¢;
4¢1
'$
	gru_inputÿÿÿÿÿÿÿÿÿ.
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_8514^$"#;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ.
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_8529^$"#;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ.
p

 
ª "ÿÿÿÿÿÿÿÿÿ¡
"__inference_signature_wrapper_8864{$"#C¢@
¢ 
9ª6
4
	gru_input'$
	gru_inputÿÿÿÿÿÿÿÿÿ."-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ