
# Sinhala dataset loader
# Unicode normalization
# ZWJ/ZWNJ handling
# Grapheme cluster validation
# Hard negative mining support
 

import numpy as np
import cv2
import os
import unicodedata
from paddle.io import Dataset
from .imaug import transform, create_operators


class SinhalaDataset(Dataset):
    
    def __init__(self, config, mode, logger, seed=None):
        super(SinhalaDataset, self).__init__()
        self.logger = logger
        self.mode = mode.lower()
        
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        
        self.delimiter = dataset_config.get('delimiter', '\t')
        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get('ratio_list', 1.0)
        
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        
        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.seed = seed
        
        # Sinhala-specific settings
        self.normalize_unicode = dataset_config.get('normalize_unicode', True)
        self.validate_graphemes = dataset_config.get('validate_graphemes', True)
        
        logger.info("Initialize Sinhala OCR dataset: %s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        
        if self.mode == 'train' and self.do_shuffle:
            self.shuffle_data_random()
        
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get('ext_op_transform_idx', 2)
        self.need_reset = True in [x < 1 for x in ratio_list]
    
    def normalize_sinhala_text(self, text):
        """
        Normalize Sinhala Unicode text:
        - Apply NFC normalization
        - Preserve ZWJ/ZWNJ for ligatures
        - Remove invalid characters
        """
        if not self.normalize_unicode:
            return text
        
        # NFC normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove control characters except ZWJ/ZWNJ
        cleaned = []
        for char in text:
            code = ord(char)
            if (0x0D80 <= code <= 0x0DFF or  # Sinhala block
                code == 0x200D or code == 0x200C or  # ZWJ, ZWNJ
                0x0030 <= code <= 0x0039 or  # Numbers
                code in [0x0020, 0x002E, 0x002C, 0x003F, 0x0021]):  # Space, punctuation
                cleaned.append(char)
        
        return ''.join(cleaned)
    
 