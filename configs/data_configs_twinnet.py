from configs import transforms_config_twinnet as transforms_config
from configs.paths_config_twinnet import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['ffhq_val'],
		'test_target_root': dataset_paths['ffhq_val'],
	},
	'cars_encode': {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_source_root': dataset_paths['cars_train'],
		'train_target_root': dataset_paths['cars_train'],
		'test_source_root': dataset_paths['cars_val'],
		'test_target_root': dataset_paths['cars_val'],
	},
	's_color_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['s_color_train'],
		'train_target_root': dataset_paths['s_color_train'],
		'test_source_root': dataset_paths['s_color_test'],
		'test_target_root': dataset_paths['s_color_test'],
	},

	's_print_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['s_print_train'],
		'train_target_root': dataset_paths['s_print_train'],
		'test_source_root': dataset_paths['s_print_test'],
		'test_target_root': dataset_paths['s_print_test'],
	},

	'h_product_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['h_product_train'],
		'train_target_root': dataset_paths['h_product_train'],
		'test_source_root': dataset_paths['h_product_test'],
		'test_target_root': dataset_paths['h_product_test'],
	}

}
