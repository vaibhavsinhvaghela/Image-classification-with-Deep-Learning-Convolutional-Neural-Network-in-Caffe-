import coremltools

caffe_model = ('caffe_model_2_iter_4000.caffemodel','deploy.prototxt')

labels = 'labels.txt'

coreml_model = coremltools.converters.caffe.convert(
		caffe_model,
		class_labels = labels,
		image_input_names = 'data'
	)

coreml_model.save('catdogclassifier4000.mlmodel')