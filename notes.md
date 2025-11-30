# NFL Big Data Bowl 2026 - Prediction
*Scott Landau* and *Patrick Astorga*

**Description**

In the Prediction Competition of the Big Data Bowl, participants are tasked with predicting player movement with the ball in the air. Specifically, the NFL is sharing data before the ball is thrown (including the Next Gen Stats tracking data), and stopping the play the moment the quarterback releases the ball. In addition to the pre-pass tracking data, we are providing participants with which offensive player was targeted (e.g, the targeted receiver) and the landing location of the pass.

Using the information above, participants should generate prediction models for player movement during the frames when the ball is in the air. The most accurate algorithms will be those whose output most closely matches the eventual player movement of each player.

**Competition specifics**
 - In the NFL's tracking data, there are 10 frames per second. As a result, if a ball is in the air for 2.5 seconds, there will be 25 frames of location data to predict.
 - Quick passes (less than half a second), deflected passes, and throwaway passes are dropped from the competition.

# Competition Specifics

## Competition Goal

We are required to implement the predict function:
```python
import pandas as pd
import polars as pl

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """
    Each batch of predictions (except the very first) must be returned within 5 minutes of the batch features being provided.
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    """
    predictions = pl.DataFrame({'x': [0.0] * len(test), 'y': [0.0] * len(test)}) # REPLACE WITH OUR MODEL PREDICTION

    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == len(test)
    return predictions
```
This function is passed as an argument to the `NFLInferenceServer()` constructor. The server hosts the evaluation for the model, and is required to be called in the submission.
```python
# must be called within 10 minutes of the notebook starting
inference_server.serve()
```
The predict function takes two inputs:
 - `test: pl.DataFrame`: Each row represents a prediction target. Each target is an xy position of a single player in a single frame (it your job to predict this). This is identified by 5 columns:
    - `id`: unique identifier for each target (game_id, play_id, nfl_id, frame_id)
    - `game_id`: Game identifier, unique (numeric)
    - `play_id`: Play identifier, not unique across games (numeric)
    - `nfl_id`: Player identification number, unique across players. (numeric)
    - `frame_id`: Frame identifier for each play, starting at 1 from the moment of release of the ball. The maximum value for a play will be the same as the num_frames_output value from the corresponding input. (numeric)
 - `test_input: pl.DataFrame`: Player tracking data for the same plays as in test, up to the moment of release of the ball. The rows/columns are in same format as the input training data.

The predict function expects one output:
 - `predictions: pl.DataFrame | pd.DataFrame`: A dataframe with the same number of rows as test, that must contain columns:
    - `id`: same unique identifier as corresponding row in test
    - `x`: predicted player position along the long axis of the field
    - `y`: predicted player position along the short axis of the field

## Input Training Data Format
The input data contains tracking data before the pass is thrown. Input training data is contained in the `input_2023_w[01-18].csv` files. Each row represents an xy position of a single player in a single frame. There are 23 columns:
 - `game_id`: Game identifier, unique (numeric)
 - `play_id`: Play identifier, not unique across games (numeric)
 - `player_to_predict`: whether or not the x/y prediction for this player will be scored (bool)
 - `nfl_id`: Player identification number, unique across players (numeric)
 - `frame_id`: Frame identifier for each play/type, starting at 1 for each game_id/play_id/file type (input or output) (numeric)
 - `play_direction`: Direction that the offense is moving (left or right)
 - `absolute_yardline_number`: Distance from end zone for possession team (numeric)
 - `player_name`: player name (text)
 - `player_height`: player height (ft-in)
 - `player_weight`: player weight (lbs)
 - `player_birth_date`: birth date (yyyy-mm-dd)
 - `player_position`: the player's position (the specific role on the field that they typically play)
 - `player_side`: team player is on (Offense or Defense)
 - `player_role`: role player has on play (Defensive Coverage, Targeted Receiver, Passer or Other Route Runner)
 - `x`: Player position along the long axis of the field, generally within 0 - 120 yards. (numeric)
 - `y`: Player position along the short axis of the field, generally within 0 - 53.3 yards. (numeric)
 - `s`: Speed in yards/second (numeric)
 - `a`: Acceleration in yards/second^2 (numeric)
 - `o`: orientation of player (deg)
 - `dir`: angle of player motion (deg)
 - `num_frames_output`: Number of frames to predict in output data for the given game_id/play_id/nfl_id. (numeric)
 - `ball_land_x`: Ball landing position position along the long axis of the field, generally within 0 - 120 yards. (numeric)
 - `ball_land_y`: Ball landing position along the short axis of the field, generally within 0 - 53.3 yards. (numeric)

## Output Training Data Format
The output data contains tracking data after the pass is thrown. The output data is only given for the players for which player_to_predict=True in the corresponding input. Output training data is contained in the `output_2023_w[01-18].csv` files. Each row represents an xy position of a single player in a single frame. There are 6 columns:
 - `game_id`, `play_id`, `nfl_id`: Same identification as training data
 - `frame_id`: Frame identifier for each play, starting at 1 from the moment of release of the ball. The maximum value for a play will be the same as the num_frames_output value from the corresponding input. (numeric)
 - `x`: Player position along the long axis of the field, generally within 0-120 yards. (**TARGET TO PREDICT**)
 - `y`: Player position along the short axis of the field, generally within 0 - 53.3 yards. (**TARGET TO PREDICT**)

# Notes #1

## Player Movement Model
In each play, we are given a subset $S$ of players to predict movement for. This subset usually contains the target receiver and a few defensive backs. Let the ball be snapped at $t = 0$, and let the ball leave the air at $t = T$. Consider one of the players in $S$. For $t \in [0, T]$, let $x_t$ represent the position of the player at time $t$. In the training data, we are given
$$x_{0}, x_{Δt}, \ldots, x_{T - Δt}, x_{T}$$
Let the ball be thrown at $t = t_{throw}$. For all $t \in (t_{throw}, T]$, we want a model $M$ which can be given $x_{0}, \ldots, x_{t - \Delta t}$ and will output a prediction $\hat x_t$.

*Fact:* There is lots of redundancy in the position data
$$x_{0}, x_{Δt}, \ldots, x_{T - Δt}, x_{T}$$
In other words, the data points are related to each other in obvious ways. I propose two ways to get rid of this redundancy when making predictions.

### Velocity Prediction
We assume that between most frames, the player position does not change drastically. Therefore
$$x_{t} \approx x_{t - \Delta t}$$
Motivated by this, let
$$\Delta x_t = x_t - x_{t - \Delta t}$$
Instead of prediciting $x_t$ directly, I propose we train our model to output a prediction $\Delta \hat x_{t}$ instead. Then the actual output of the model can be computed as
$$\hat x_t = x_{t - \Delta t} + \Delta \hat x_{t}$$
In this case, the redundancy of the data is baked in to the model output for free. $\Delta x_t$ is approximately proportional to the velocity of the player at time $t$. If the model outputs 0, then the player remains stationary.

### Acceleration Prediction
We assume that between most frames, the player position and velocity does not change drastically. Therefore
$$x_{t} \approx x_{t - \Delta t} + \Delta x_{t - \Delta t}$$
Motivated by this, let
$$\Delta^2 x_t = x_t - (x_{t - \Delta t} + \Delta x_{t - \Delta t})$$
Instead of prediciting $x_t$ directly, I propose we train our model to output a prediction $\Delta^2 \hat x_t$ instead. Then the actual output of the model can be computed as
$$\hat x_t = (x_{t - \Delta t} + \Delta x_{t - \Delta t}) + \Delta^2 \hat x_t$$
In this case, the redundancy of the data is baked in to the model output for free. $\Delta^2 x_t$ is approximately proportional to the acceleration of the player at time $t$. If the model outputs 0, then the player moves in a straight line with constant speed.

## Model Inputs
While the ball is in the air, only the position data of the players in $S$ is available in the training data. Therefore I propose that it is most reasonable to restrict our model to only predicting the movement of players in $S$. At each time step, the model should make predictions for all of the players in $S$ simultanouesly, since the **past/current positions of all the players in $S$** should be passed into the model as input.

What about the position data of all the players not in $S$? Since my proposal states that we do not predict thier movement while the ball is in the air, we do not have access to thier complete history of past/current positions. However, they certainly encode important aspect about the play, such as defensive formation, offensive play type, etc. I propose that using all of the pre-throw data available as an input, we calculate a **static representation $Z$ which encodes valuable information about the play**, and pass $Z$ as an input to the model at all time steps.

*Note: By encoding the position data of the players not in $S$ into a static representation $Z$, I am making an assumption that these players are not actively influencing the play while the ball is in the air. I still have not investigated in depth how the subset $S$ is selected.*

A **time encoding $\tau$** is a neccesary input to the model, since the ball exits the air after a set number of frames, and the player positions need to be updated accordingly.

Finally, important data from the dataset such as `play_direction`, `player_role`, `ball_land_x`, and `ball_land_y` should be passed to the model as inputs.

# Notes #2
## Transformer
Introduced in the 2017 paper [Attention is All You Need]. This model is charecterized by its use of **Attention** to handle relationships between two sequences of embeddings. The Attention layer is processed in parallel on all embeddings, allowing each one to attend to all the others simultaneously. This saves computational cost and allows for long-range dependencies to be learned directly.

### Attention
For $Q \in \mathbb{R}^{m \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, and $V \in \mathbb{R}^{n \times d_v}$, the Attention function is as follows
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
Each row of $Q$ is a "query" computed from an embedding in sequence 1, and each row of $K$ and $V$ is a "key" and "value" computed from an embedding in sequence 2, respectively. The softmax function is computed row-wise. The $i$th row of the output is a weighted sum of the rows of $V$ with weights associated with the  similarity between the $i$th row of $Q$ and the corresponding rows of $K$.

**Example:** Each query attends to the $K$ and $V$ matrices independantly, so to illustrate how this function works, we consider its output on a single query $\vec q$. Hence $Q = \vec q^T$. Let
$$ K = \left[ \vec k_1 \,\cdots\, \vec k_n \right]^T, \quad V = \left[ \vec v_1 \,\cdots\, \vec v_n \right]^T $$

1. The matrix multiplication $QK^T$ computes the dot product between the query $\vec q$ and each of the keys $\vec k_1, \ldots, \vec k_n$
$$ QK^T = \vec q^T\left[ \vec k_1 \,\cdots\, \vec k_n \right] = \left[ \vec q^T\vec k_1 \,\cdots\, \vec q^T\vec k_n \right] $$

2. For numerical stability, the dot products are scaled down by the square root of the dimensionality of the queries and keys. This is to prevent these values fromgrowing too large when the number of dimensions is increased.

3. These dot products are a measure of similarity between the query and each of the keys. We want the output to be a weighted sum of the values $\vec v_1, \ldots, \vec v_n$, where the weights are scaled accordingly to the relative similarty of the corresponding keys $\vec k_1, \ldots, \vec k_n$ to the query $\vec q$. We want all of the weights to be nonnegative and to be normalized so thier sum is equal to 1. We accomplish this with the softmax function
$$ \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) = \left[ w_1 \,\cdots\, w_n \right] $$

4. The final matrix multiplcation by $V$ simply takes the weighted sum of the values
$$ \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V = \left[ w_1 \,\cdots\, w_n \right]\left[ \vec v_1 \,\cdots\, \vec v_n \right]^T = w_1\vec v_1^T + \cdots + w_n\vec v_n^T $$

When $Q$ contains the queries of an entire sequence of embeddings, you can imagine this operation being computed row-wise: i.e. each row of the output is the result of the corresponding row of $Q$ (the query). So the Attention operation allows each embedding in one sequence to recieve information from every embedding in another sequence. When the two sequences are the same, we call this **Self-Attention**. When the two sequences are different, we call this **Cross-Attention**. Many times, we compute the result of many Attention layers in parrallel, and concatanate/add thier outputs. This is called **Multi-Head Attention**.

### Transformer model
The most basic transformer model takes in one or more sequences of embeddings, and alternates between applying Attention blocks on the entire sequences with applying Multi-Layer Perceptrons (MLPs) on the individual embeddings. The operations usually have skip connections between the inputs and outputs. Hence the inputs and outputs of each layer typically have the same dimensionality, commonly referred to as the **model dimensionality** or $d_{model}$. After all of the Attention/MLP blocks, the output is a transformed version of the original sequence.

If the model is an **encoder**, then this transformed sequence is used a representation of the input sequence, to be used by other models.

If the model is a **decoder**, then the transformed embeddings are fed into a final MLP which outputs an answer for some practical use. For example, the output embeddings of GPT are used to predict the next word in a passage of text.

## Big Data Bowl Architecture #1

Encoder-Decoder Tranformer model, motivated by the core problem that we do not have access to ground truth post-throw tracking data for players not in $S$. We rely on the Encoder to model interactions between the full set of players PRE-THROW to capure valuable information about the play. We then use that information to assist the decoder which predicts the motion of the players in $S$ POST-THROW.

Therefore, the representations of the players not in $S$ are effectively static after the throw. Obviously these features are not actually static in reality. However, I dont see it reasonable to try to predict thier movement after the throw, since we do not have ground truth. I think that treating them as static features is one possible get-around to this problem. In fact, I'd identify this as the **core idea underlying this model**. If we decide that this is unreasonable, then this model cannot work.

### Encoder
The encoder acts upon the pre throw tracking data. Suppose that there are $A$ total players in the play, and there are $T_{pre}$ frames of pre-throw tracking data. In the input layer of the decoder, each of the $A \times _{pre}$ available inputs are embedded into a $D$-dimensional embedding space. Hence the Attention blocks of the encoder operate on a tensor of shape $(A, T_{pre}, D)$. The encoder consists of repeated blocks of Factorized Self-Attention (without any masking). Inspired by [Scene Transformer] Sec 3.3, after the first block of Factorized Self-Attention, we compute the mean of the feature tensor along the time dimension and append it to the time dimension as an additional feature, not tied to any time step. The final shape of the feature tensor is hence $(A, T_{pre} + 1, D)$. The output of the encoder is the feature tensor sliced along the additional time dimension, a tensor of shape $(A, D)$.

### Decoder
The decoder acts upon the tracking data of the players in $S$. Suppose that there are $P$ players in $S$ and $T_{post}$ total frames of of post-throw tracking data. In the input layer of the decoder, each of the $P \times (T_{pre} + T_{post})$ available inputs are embedded into a $D$-dimensional embedding space. Hence the Attention blocks of the decoder operate on a tensor of shape $(P, T_{pre} + T_{post}, D)$. The decoder consists of repeated blocks of Factorized Self-Attention and Cross-Attention (to the Encoder output). The output layer of the decoder maps each $D$ dimensional embedding to an $xy$-coordinate which predicts the position in the next frame. Hence the output of the decoder is a tensor of shape $(P, T_{pre} + T_{post}, 2)$.

### Methods
1. **Position Encodings:** The entire model will use a shared coordinate system. The origin will be set to be one of {Ball landing location, Quarterback position, Center line of scrimmage, etc.}. Reference: [Scene Transformer] Sec 3.1. The coordinate system will be rotated to be aligned to foward progress. The $xy$-coordinates will be mapped into a high dimensional embedding using a MLP. Reference: [FootBots] Sec 3.3, [TranSPORTmer] Sec 4.2.

2. **Attribute Encodings:** Player attributes (such as offense/defense, receiver/catcher, etc.) can be appended to the position data before being fed into the MLP embedding network. Reference: [TranSPORTmer] Sec 4.2

3. **Temporal Encodings:** When embeddings span across time, a sinusoidal temporal encoding $\tau$ should be added to the position/attribute encoding. Reference: [Attention is All You Need] Sec 3.5

1. **Factorized Self-Attention:** When the sequence spans over the position and time axes, Self-Attention will not consider over the entire sequence at once. Instead, alternating blocks of Self-Attention will act over the space and time axes individually. Reference: [Scene Transformer] Sec 3.2

1. **Masking:** Since the decoder is autoregressive, we apply causal masking during Self-Attention along the time axis. Reference: [Attention is All You Need] Sec 3.2.3. If we want to enforce that the final position of the targeted receiver will always be equal to the football landing location, then we can use the Goal-Conditioned Predicition (GCP) masking strategy, which is the same as before except we allow the target receiver's embeddings to attend to it's final location, which we set to equal the landing location of the ball. Reference: [Scene Transformer] Figure 2.

### Discussion
Do we encode static features about the field such as the sidelines? I feel like we can get away without encoding information about the sidelines, since they are inherently encoded in the training data.

How do we incorperate ball landing position into this model? We can use Goal Conditioned Prediction (GCP) to enforce that the model predicts the receiver to end its route at the ball landing location. This certainly allows the model to be simpler. However, this assumption does not hold true in the training data, so this assumption is inaccurate. Other methods such as appending the landing location as an artificial agent should be considered.

### Model size and hyperparameters
- Considerations on model size must be made based on computational resources.

- Given a model architecture, can we compute the memory footprint? This will allow us to compute an upper bound on model size. Its necessary to know the upper/lower limits of player/frame counts, which determine the required context size.

- Are their standards for relative parameter counts between the Encoder and Decoder?

- Should we use regularization strategies such as dropout?

[Attention is All You Need]: https://arxiv.org/abs/1706.03762
[Scene Transformer]: https://arxiv.org/abs/2106.08417
[FootBots]: https://arxiv.org/abs/2406.19852
[TranSPORTmer]: https://arxiv.org/abs/2410.17785

# Notes #3
Clarification:
We will forecast post-throw trajectories for all players in S for each frame until the ball arrives.

We are predicting change in acceleration squared, which can be integrated into x.

Normalize the field such that the offense always goes left to right to create symmetry breaking constraint.

Embed positions into a learnable vector.

Don't hard code out of bounds because player sometimes do go out of bounds. Rather, I propose creating a distance to boundary feature as sideline awareness so that the model doesn't put players in the stands.

# Create tensor shapes
Pre-throw: T_pre frames and A total players on the field. Encoder input (A,T_pre,D_in) embeds to (A,T_pre,D_model)

Decoder: A players to predict. Decoder input (P,T_post,D_model) output (P,T_post,2).

# Adjustment to 2D
Alternate between self-attention over time and self-attention over players. Attend across time axis in the first and across agents in the second.

Each block will be Attention -> residual -> LayerNorm -> MLP -> residual -> LayerNorm

Encoder has no mask but decoder's self attention over time will use a mask for future frames. Over players and cross attention will ahve no mask.

#GCP
For the targeted receiver only, the final frame is unmasked to include the final position. Earlier steps remain masked.

#Encoder output for players not in S
Encoder output: (A,D_model) after factorized blocks and scene summary

For players not in S, they are never decoded post-throw, but their encoder embeddings represent pre-throw context, showing formations and plays

For players in S:
Can either use a separate S-only decoder or initialize the first-frame embeddings for each player in S as the ecoder's corresponding player vector. Note that we are adding the ball as an extra agent in the encoder and decoder.

#Loss
Frame-wise Huber or MSE on positions, summed over players in S and predicted timesteps. Can also add velocity smoothness to discourage high accelerations. Can also weight targeted receiver more heavily.

#Data coding preparation
Train split by game to avoid leakage from one game to another.

Padding strategy since we have variable inputs and transformers need fixed-shape tensors.

Sequence: ordered data for one play

Batch: collection of plays that are processed in parallel together. Transformers require uniform tensor lengths across a batch.

Transformers can mathematically handle variable-length sequences, but PyTorch cannot batch variable-length tensors. I propose padding player dimensions to max players in each batch and using a binary player_mask for valid vs padded players, which will be used in the self-attention space axis.The same process is used in time-dimension to fit the max_frames.

Only compute loss up to true in-air duration.


Pytorch:

MLPEmbed: projects raw features to the model

PositionalEncodingTime: sinusoidal

PositionalEncodingSpace: 2D MLP from position coordinate to embedding

FactorizedAttentionBlock: Time-attention and space-attention sublocks

Encoder: stack of FactorizedAttentionBlocks to (A,D_model)

DecoderBlock: Time and space self-attention, cross-attention to encoder output, MLP

Decoder: Stack of DecoderBlocks embedded to (P,T_post,2)

MotionModel: Combines encoder and decoder, handles masks, shapes, and loss

#Questions
Do we commit to acceleration prediction?

How big should D_model, num_heads, num_layers be given our GPU limits?

Do we use GCP in the v1 or just feed ball landing as an extra agent?

Do we treat the ball as an agent in encoder, decoder, both, or neither?


Max boundaries found from code below:

GLOBAL TRAINING STATS:

max_P: 9

max_A: 17

max_T_pre: 115

max_T_post: 94


#Order of operations
Encoder (choose one):

space_attn -> MLP -> time_attn -> MLP

time_attn -> space_attn -> MLP

#Version plans
v1: constant velocity model → sanity check

v2: single-layer transformer

v3: factorized transformer with encoder/decoder

v4: add GCP, attribute embeddings, ball agent, acceleration prediction

# Notes #4

I feel like it is not too much of a stretch to say that the approach of fetching the tracking data from the CSV and converting it to a Tensor-ready format inside of the `__getitem___` function is way to slow, since the dataframe lookups are $O(n)$ and there are a bunch of python loops used to populate the numpy arrays.

Instead we do this step as preprocessing and generate the python object
```
[
    {
        "game_id": int,
        "play_id": int,
        "pre_xy": np.array([...]),        # shape (P, T_pre, 2)
        "post_xy": np.array([...]),       # shape (A, T_post, 2)
        "roles": np.array([...]),         # shape (P,)
        "roles_S": np.array([...]),       # shape (A,)
        "ball_landing": np.array([x, y]), # shape (2,)
    },
    ...
]
```
which can then be serialized and stored to the disk using pickle. Since the array allows for $O(1)$ lookups and the data is already stored in Tensor-ready NumPy arrays, the CPU workload should be greatly reduced for the workers of the dataloader.

