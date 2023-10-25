import json
import copy
import six
import logging


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size=1,
                 hidden_size=16,
                 num_hidden_layers=3,
                 num_attention_heads=1,
                 intermediate_size=32,
                 act_fn="gelu",
                 initializer_range=0.02,
                 dropout=0.1,
                 hidden_dropout_prob=0.1,
                 attn_dropout=0.1,
                 seq_len = 400,
                 conv_size = 64
):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout_prob
        self.act_fn = act_fn
        self.attn_dropout = attn_dropout
        self.initializer_range = initializer_range
        self.dropout = dropout
        self.seq_len = seq_len
        self.conv_size = conv_size

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        """从json配置文件读取配置信息"""
        with open(json_file, 'r') as reader:
            text = reader.read()
        logging.info(f"成功导入BERT配置文件 {json_file}")
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
