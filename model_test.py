import os
import logging
import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import argparse

SEQ_LEN     = 64
CHANNELS    = 543 * 3
PAD         = 0.0
NUM_CLASSES = 250


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # INFO и ниже скрыты
logging.getLogger('absl').setLevel(logging.ERROR)

# ─── Параметры командной строки ─────────────────────────────────
parser = argparse.ArgumentParser(description='ASL recognition demo')
parser.add_argument('--camera', type=int, default=0,
                    help='Index of the camera to use (default: 0)')
args = parser.parse_args()

# ─── Глобальные константы ───────────────────────────────────────
SEQ_LEN     = 64
CHANNELS    = 543 * 3
PAD         = 0.0

# ─── Загружаем словарь «жест→индекс» и инвертируем ──────────────
with open('sign_to_prediction_index_map.json', 'r', encoding='utf-8') as f:
    sign2idx = json.load(f)
inv_sign2idx = {v: k for k, v in sign2idx.items()}
NUM_CLASSES = len(sign2idx)



# ─── 1) Загрузка модели ─────────────────────────────────────────────
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    qkv одним Dense без bias
    scale = 1/√d_k
    Dropout на attention score
    маска накладывается как –1e9 перед softmax
    """
    def __init__(self, dim=256, num_heads=4, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.dim        = dim
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv  = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, x, mask=None):                      # x: (B,T,C)
        B, T = tf.shape(x)[0], tf.shape(x)[1]

        qkv = self.qkv(x)                              # (B,T,3*dim)
        qkv = tf.reshape(qkv, [B, T, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])       # (3,B,H,T,D)
        q, k, v = tf.unstack(qkv, 3, axis=0)           # (B,H,T,D) ×3

        attn = tf.matmul(q, k, transpose_b=True) * self.scale  # (B,H,T,T)

        if mask is not None:                           # mask: (B,T)
            mask = mask[:, None, None, :]              # (B,1,1,T)
            attn = tf.where(mask, -1e9, attn)          # set –∞

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.drop(attn)

        x = tf.matmul(attn, v)                         # (B,H,T,D)
        x = tf.transpose(x, [0, 2, 1, 3])              # (B,T,H,D)
        x = tf.reshape(x, [B, T, self.dim])            # (B,T,C)

        return self.proj(x)


def TransformerBlock(dim=256, num_heads=4, expand=4,
                     attn_dropout=0.2, drop_rate=0.2,
                     activation='swish'):
    def block(inputs):
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(inputs)
        x = MultiHeadSelfAttention(dim, num_heads, attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim*expand, activation=activation, use_bias=False)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        return tf.keras.layers.Add()([attn_out, x])
    return block

class CausalDWConv1D(tf.keras.layers.Layer):
    """Depth-wise 1-D conv с каузальным паддингом — 1 в 1 как у автора."""
    def __init__(self, ksize=17, **kw):
        super().__init__(**kw)
        self.ksize = ksize
        self.dw = tf.keras.layers.DepthwiseConv2D(
            (ksize, 1), padding='valid', use_bias=False)

    def call(self, x):
        # x: [B, T, C]
        x = tf.pad(x, [[0, 0], [self.ksize - 1, 0], [0, 0]])  # только слева
        x = tf.expand_dims(x, 2)            # → [B, T+ksize-1, 1, C]
        x = self.dw(x)                      #      высота снова T
        return tf.squeeze(x, 2)             # ← [B, T, C]


class ECA(tf.keras.layers.Layer):
    """Efficient Channel Attention (авторская реализация)."""
    def __init__(self, k_size=3, **kw):
        super().__init__(**kw)
        self.conv = tf.keras.layers.Conv1D(1, k_size, padding='same',
                                           use_bias=False)

    def call(self, x):
        y = tf.reduce_mean(x, axis=1, keepdims=True)  # GAP
        y = self.conv(y)
        y = tf.keras.activations.sigmoid(y)
        return x * y

def Conv1DBlock(dim, ksize, drop_rate=0.2):
    """MBConv-подобный блок, 1-в-1 как у автора."""
    def block(inputs):
        x = tf.keras.layers.Dense(dim*2, activation='swish', use_bias=False)(inputs)
        x = CausalDWConv1D(ksize)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = ECA()(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        return tf.keras.layers.Add()([inputs, x])
    return block

class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate=0.5, start_step=0, **kw):
        super().__init__(**kw)
        self.rate  = rate
        self.start = start_step
        self.do    = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None):
        if training:
            step = tf.summary.experimental.get_step()
            # если счётчик шагов ещё не задан — считаем, что «рано»,
            # и просто не применяем dropout
            if step is None or step < self.start:
                training = False
        return self.do(x, training=training)


def get_model(max_len=64, dropout_step=0, dim=192):
    inp = tf.keras.Input((max_len, CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD)(inp)

    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)

    for _ in range(3):
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    for _ in range(3):
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    if dim == 384:               # «4×-model» у автора
        for _ in range(3):
            x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)
        for _ in range(3):
            x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.Dense(dim*2, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, name='classifier')(x)
    return tf.keras.Model(inp, out)


model = get_model(max_len=64, dim=192)
model.load_weights('epoch-18_valAcc-0.735.h5')

# ─── MediaPipe и буфер последовательности ─────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
buffer = deque(maxlen=SEQ_LEN)

def get_landmarks_or_zero(landmarks, count):
    if landmarks:
        return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
    else:
        return [[0.0, 0.0, 0.0]] * count

# ─── Запускаем реальное время с камеры ────────────────────────────
cap = cv2.VideoCapture(args.camera)
frame_id = 0

last_label = ''
last_prob  = 0.0

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # зеркало + RGB для MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        res = holistic.process(img_rgb)
        img_rgb.flags.writeable = True
        image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # рисуем ключевые точки
        mp_drawing.draw_landmarks(image, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1))
        mp_drawing.draw_landmarks(image, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2))
        mp_drawing.draw_landmarks(image, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(121,22,76), thickness=2))
        mp_drawing.draw_landmarks(image, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2))

        # собираем координаты 543 точек
        coords = []
        coords += get_landmarks_or_zero(res.face_landmarks,    468)
        coords += get_landmarks_or_zero(res.pose_landmarks,     33)
        coords += get_landmarks_or_zero(res.left_hand_landmarks,21)
        coords += get_landmarks_or_zero(res.right_hand_landmarks,21)
        arr = np.array(coords, dtype=np.float32).flatten()  # (1629,)
        buffer.append(arr)

        # раз в 4 кадра, когда накопили SEQ_LEN, предсказываем
        if len(buffer) == SEQ_LEN and frame_id % 4 == 0:
            clip   = np.stack(buffer, axis=0)[None, ...]  # (1,64,1629)
            inp    = clip * 2.0 - 1.0
            logits = model.predict(inp, verbose=0)
            probs  = tf.nn.softmax(logits, axis=-1).numpy()[0]
            idx    = np.argmax(probs)
            last_label = inv_sign2idx[idx]
            last_prob  = probs[idx]

        h, w = image.shape[:2]
        panel_height = 40
        cv2.rectangle(
            image,
            (0, h - panel_height),
            (w, h),
            (0, 0, 0),       # чёрная заливка
            thickness=-1     # -1 = fill
        )

        # 2) Выводим белый текст на этой панели
        if last_label:
            text = f'{last_label}: {last_prob*100:.1f} %'
            cv2.putText(
                image,
                text,
                (10, h - 10),               # отступ от левого и нижнего края панели
                cv2.FONT_HERSHEY_SIMPLEX,
                1,                           # масштаб шрифта
                (255, 255, 255),             # белый цвет
                2,                           # толщина линий
                cv2.LINE_AA
            )

        cv2.imshow('ASL Recognition', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()