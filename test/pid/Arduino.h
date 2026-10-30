#pragma once
template<class T> T constrain(T x,T lo,T hi){return x<lo?lo:x>hi?hi:x;}
