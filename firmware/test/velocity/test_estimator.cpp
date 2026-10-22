#include "../../src/control/VelocityEstimator.h"
#include <cassert>
#include <cmath>
#include <limits>
int main() {
    VelocityEstimator v;
    assert(v.update(12,0,0.005f,165,0.2f,true)==12);
    float x=0;
    for(int i=0;i<200;i++) x=v.update(12,0,0.005f,165,0.2f,true);
    assert(std::fabs(x-12)<0.001f);
    x=v.update(12,1,0.005f,165,0.2f,true); assert(x>12);
    for(int i=0;i<1000;i++) x=v.update(12,1,0.005f,165,0.2f,true);
    assert(x<36); // constant acceleration bias produces bounded error
    for(int i=0;i<1000;i++) x=v.update(12,0,0.005f,165,0.2f,true);
    assert(std::fabs(x-12)<0.001f);
    assert(v.update(5,10,0.005f,165,0,true)==5);
    assert(v.update(6,10,0.005f,0,0.2f,true)==6);
    assert(v.update(6,10,0.005f,165,0.2f,false)==0);
    assert(v.update(-8,10,0.005f,165,0.2f,true)==-8);
    assert(v.update(4,std::numeric_limits<float>::quiet_NaN(),0.005f,165,0.2f,true)==4);
    VelocityEstimator a,b;
    a.update(0,0,.005f,165,.2f,true); b.update(0,0,.01f,165,.2f,true);
    float av=0,bv=0;
    for(int i=0;i<200;i++)av=a.update(0,1,.005f,165,.2f,true);
    for(int i=0;i<100;i++)bv=b.update(0,1,.01f,165,.2f,true);
    assert(std::fabs(av-bv)<.001f);
}
