from keras.models import Model, Sequential
from keras.layers import SeparableConv2D
from keras import regularizers
import json

model = Sequential()
model.add(SeparableConv2D(32, 3, depthwise_regularizer=regularizers.l2(0.01),
                          pointwise_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l2(0.01),
                          activity_regularizer=regularizers.l2(0.01), depthwise_constraint='max_norm',
                          bias_constraint='max_norm', pointwise_constraint='max_norm',
                          activation='relu', input_shape=(16, 16, 1)))

json_string = Model.to_json(model)
with open("sep3d_3.json", "w") as f:
	json.dump(json.loads(json_string), f, indent=4)

#with open("keras_export_test.json", "r") as f:
#	response = json.load(f)
#	response = json.dumps(response['net'])
#	with open("sep3d_3.json", "w") as f:
#		json.dump(json.loads(response), f, indent=4)
