import tensorflow as tf
print(tf.__version__)   # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus):
    print("쥐피유 돈다!!!")
else:
    print("쥐피유 없다! xxxx")

# CPU:
# GPU: 
    
    
    