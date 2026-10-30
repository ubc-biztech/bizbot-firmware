Run the host-side PID checks from the repository root:

```sh
c++ -std=c++11 -I test/pid src/control/PID.cpp test/pid/test_derivative_boost.cpp -o /tmp/bizbot-pid-test
/tmp/bizbot-pid-test
```

The local Arduino header supplies only `constrain` for this host test. Checks cover baseline D, symmetric and increasing angle response, half-boost angle, target-relative error, rate direction, invalid settings, disabled boost, and output saturation.
