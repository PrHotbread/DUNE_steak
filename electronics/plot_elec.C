double coldelec(double time)//, double gain, double shaping)
{
  double gain = 15.0; // 14.0 mV/fC
  double shaping = 4.0; // or 2.2 microseconds in LAr
  
  if (time <=0 || time >= 10){ // * units::microsecond) { // range of validity
  return 0.0;
  }
  
  const double reltime = time/shaping;
  // a scaling is needed to make the anti-Lapalace peak match the expected gain
  // fixme: this scaling is slightly dependent on shaping time. See response.py
  gain *= 10*1.012;
  return 4.31054*exp(-2.94809*reltime)*gain
  -2.6202*exp(-2.82833*reltime)*cos(1.19361*reltime)*gain
  -2.6202*exp(-2.82833*reltime)*cos(1.19361*reltime)*cos(2.38722*reltime)*gain
  +0.464924*exp(-2.40318*reltime)*cos(2.5928*reltime)*gain
  +0.464924*exp(-2.40318*reltime)*cos(2.5928*reltime)*cos(5.18561*reltime)*gain
  +0.762456*exp(-2.82833*reltime)*sin(1.19361*reltime)*gain
  -0.762456*exp(-2.82833*reltime)*cos(2.38722*reltime)*sin(1.19361*reltime)*gain
  +0.762456*exp(-2.82833*reltime)*cos(1.19361*reltime)*sin(2.38722*reltime)*gain
  -2.620200*exp(-2.82833*reltime)*sin(1.19361*reltime)*sin(2.38722*reltime)*gain
  -0.327684*exp(-2.40318*reltime)*sin(2.5928*reltime)*gain +
  +0.327684*exp(-2.40318*reltime)*cos(5.18561*reltime)*sin(2.5928*reltime)*gain
  -0.327684*exp(-2.40318*reltime)*cos(2.5928*reltime)*sin(5.18561*reltime)*gain
  +0.464924*exp(-2.40318*reltime)*sin(2.5928*reltime)*sin(5.18561*reltime)*gain;
}

void plot_elec(){
// TF1* f = new TF1("f","coldelec(x) * 3.43 * 4095 / 1400",-5,15); // 4095 ADC to 1400 mV conversion,
                                                                // one step in pulser is 3.43 fC  ̃ 21.4 ke
TF1* f = new TF1("f","coldelec(x)",-5,15);

f->SetNpx(5000);
f->Draw();
}

