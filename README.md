
# Ego4D Hands & Objects Benchmark Clip from i3dresnet



## Task Definition

The Hands & Objects benchmark captures how the camera-wearer changes the state of an object by using or manipulating it – which we call an object state change. Though cutting a piece of lumber in half can be achieved through many methods (e.g., various tools, force, speeds, grasps, end effectors), all should be recognized as the same state change.  

Object state changes can be viewed along temporal, spatial, and semantic axes, leading to these three tasks:

1. [Point-of-no-return temporal localization](./state-change-localization-classification/README.md): given a short video clip of a state change, the goal is to estimate the keyframe that contains the point-of–no-return (PNR) (the time at which a state change begins)

1. [State change object detection](./state-change-localization-classification/README.md): given three temporal frames (pre, post, PNR), the goal is to regress the bounding box of the object undergoing a state change

1. [Object state change classification](./state-change-localization-classification/README.md): Given a short video clip, the goal is to classify whether an object state change has taken place or not

Please see the individual README for each of the sub-task directories for more detail. 

### License

Ego4D is released under the MIT License.
