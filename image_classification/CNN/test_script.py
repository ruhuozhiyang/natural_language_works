# import sys
#
# if __name__ == '__main__':
#     if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
#         raise ValueError("""usage: python run_cnn.py [train / test]""")

import tensorflow as tf1

tf = tf1.compat.v1
tf.disable_v2_behavior()
sess = tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True))
a=tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b=tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c=tf.matmul(a, b)
print(sess.run(c))
