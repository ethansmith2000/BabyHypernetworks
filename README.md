# BabyHypernetworks

This repository contains a bunch of fun concepts for neural network layers that center around projecting an input into weights to be used as a transformation for another (or the same) input

I like to think of common layers such as AdaNorm or Attention as "baby" hypernetworks. 
As opposed to conventional hypernetworks which may create weights for an entire network, these layers generate simple operations such as elementwise scale/shift and a linear transformation respectively.
This is cool, we can adapt the flow of our network dynamically according to our inputs.
Following this framework, what if we could create more complex operations while still staying in the realm of "baby" sized hypernetworks?

Accompanying post: https://sweet-hall-e72.notion.site/Why-are-Modern-Neural-Nets-the-way-they-are-And-Hidden-Hypernetworks-6c7195709e7b4abbada921875a951c54?pvs=4

thanks to rami_mmo for fun discussions
