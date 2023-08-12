import tensorflow as tf
import numpy as np
import cv2

# Load VGG19 model pretrained on ImageNet
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Content and style layers for style transfer
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def preprocess_image(image_path):
    # image = cv2.imread(image_path)
    image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = image[tf.newaxis, :]
    return image

def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def get_content_model():
    content_model = tf.keras.models.Model(
        vgg.input, outputs=[vgg.get_layer(name).output for name in content_layers]
    )
    return content_model

def get_style_model():
    style_model = tf.keras.models.Model(
        vgg.input, outputs=[vgg.get_layer(name).output for name in style_layers]
    )
    return style_model

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_transfer(content_image, style_image, num_iterations=10, content_weight=1e2, style_weight=30):
    content_image = preprocess_image(content_image)
    style_image = preprocess_image(style_image)
    
    content_model = get_content_model()
    style_model = get_style_model()

    target_content = content_model(content_image)
    target_style = style_model(style_image)

    generated_image = tf.Variable(content_image)

    optimizer = tf.optimizers.Adam(learning_rate=0.03, beta_1=0.99, epsilon=1e-1)

    for iteration in range(num_iterations):
        with tf.GradientTape() as tape:
            gen_content = content_model(generated_image)
            gen_style = style_model(generated_image)

            content_loss = tf.reduce_mean(tf.square(gen_content - target_content))

            style_loss = 0
            for target_gram, gen_gram in zip(target_style, gen_style):
                style_loss += tf.reduce_mean(tf.square(gram_matrix(gen_gram) - gram_matrix(target_gram)))
            style_loss /= num_style_layers
            
            total_loss = content_weight * content_loss + style_weight * style_loss

        gradients = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        generated_image.assign(clip_0_1(generated_image))
        print(f"Iteration: {iteration + 1}, Total Loss: {total_loss.numpy()}")

    return generated_image.numpy()

# content_image_path = "content.png"
# style_image_path = "style.png"
# output_image = style_transfer(content_image_path, style_image_path)

# output_image = np.squeeze(output_image, axis=0)
# output_image = (output_image * 255).astype(np.uint8)
# output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# cv2.imshow("Cartoonized Image", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("cartoonized_image.jpg", output_image)
