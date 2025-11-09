# NFL Big Data Bowl 2026 - Prediction

*Scott Landau* and *Patrick Astorga*

**Description**

In the Prediction Competition of the Big Data Bowl, participants are tasked with predicting player movement with the ball in the air. Specifically, the NFL is sharing data before the ball is thrown (including the Next Gen Stats tracking data), and stopping the play the moment the quarterback releases the ball. In addition to the pre-pass tracking data, we are providing participants with which offensive player was targeted (e.g, the targeted receiver) and the landing location of the pass.

Using the information above, participants should generate prediction models for player movement during the frames when the ball is in the air. The most accurate algorithms will be those whose output most closely matches the eventual player movement of each player.

**Competition specifics**

 - In the NFL's tracking data, there are 10 frames per second. As a result, if a ball is in the air for 2.5 seconds, there will be 25 frames of location data to predict.
 - Quick passes (less than half a second), deflected passes, and throwaway passes are dropped from the competition.
