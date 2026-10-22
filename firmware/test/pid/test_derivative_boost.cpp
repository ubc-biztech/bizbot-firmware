#include "../../src/control/PID.h"
#include <cassert>
#include <cmath>
#include <limits>
static void near(float a,float b){assert(std::fabs(a-b)<0.00001f);}
int main(){
    PID directional(0.04f,0,0);
    directional.setDirectionalP(0.04f,0.06f);
    near(directional.update(0,2,0.005f),-0.08f);
    near(directional.update(0,-2,0.005f),0.12f);
    near(directional.update(3,2,0.005f),0.06f);
    near(directional.update(2,2,0.005f),0);
    directional.setProportionalBoost(0.65f,0.5f);
    near(directional.update(0,1,0.005f),0.04f*(-1-0.65f*std::tanh(2.0f)));
    near(directional.update(0,-1,0.005f),0.06f*(1+0.65f*std::tanh(2.0f)));
    directional.setTunings(0.03f,0,0);
    directional.setProportionalBoost(0,0.5f);
    near(directional.update(0,1,0.005f),-0.03f);
    near(directional.update(0,-1,0.005f),0.03f);
    PID pid(0,0,0.001f);
    pid.setDerivativeBoost(1,5);
    pid.update(0,0,0.005f,10); near(pid.lastDTerm(),-0.01f);
    pid.update(0,5,0.005f,10); near(pid.lastDTerm(),-0.015f);
    pid.update(0,-5,0.005f,10); near(pid.lastDTerm(),-0.015f);
    pid.update(0,10,0.005f,10); near(pid.lastDTerm(),-0.018f);
    pid.update(0,1000,0.005f,10); assert(pid.lastDTerm()>-0.020001f);
    pid.update(5,5,0.005f,10); near(pid.lastDTerm(),-0.01f);
    pid.update(0,5,0.005f,-10); near(pid.lastDTerm(),0.015f);
    pid.setDerivativeBoost(2,0); // rejected: preserve previous valid settings
    pid.setDerivativeBoost(std::numeric_limits<float>::quiet_NaN(),5);
    pid.update(0,5,0.005f,10); near(pid.lastDTerm(),-0.015f);
    pid.setDerivativeBoost(0,5);
    pid.update(0,20,0.005f,10); near(pid.lastDTerm(),-0.01f);
    pid.setDerivativeBoost(4,5); pid.setOutputLimits(-0.02f,0.02f);
    near(pid.update(0,20,0.005f,100),-0.02f);
}
