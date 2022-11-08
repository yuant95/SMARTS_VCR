# Explanation

1. Smoothed bezier curve used for path planning from the waypoints ahead of the ego car;
2. A trained neural network is used for classfying the action into 6 categories of collision, wrong way, on shoulder, off road, off route. The data is collected offlined. The input includes a bird view 256 x 256 image, ego position, ego heading, action, the future 5 waypoints. For the network architecture, see the code in mnn.py. It is a neural network combined with a CNN and MLP for the mixed input data. 
3. Each step, a action will be proposed from the bezier curve with speed of 1, then use the classfier to score the action based on how safe it is. 
4. If it is not safe, currently a hack is used to decided whether the ego car needs to accelerate or decelerate by calculating the surrounding obstacles. If no obstacles in the front, acclerate, otherwise decelerate. This should be replaced by the proper work of the NN we trained earilier. But I ran out of time to finish this part. 
5. Repeat. 

GitHub link: https://github.com/yuant95/SMARTS_VCR