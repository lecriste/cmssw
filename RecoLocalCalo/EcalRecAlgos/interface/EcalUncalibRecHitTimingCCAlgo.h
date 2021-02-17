#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitTimingCCAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitTimingCCAlgo_HH

/** \class EcalUncalibRecHitTimingCCAlgo
  *  CrossCorrelation algorithm for timing reconstruction
  *
  *  \author N. Minafra, J. King, C. Rogan
  */

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes.h"

class EcalUncalibRecHitTimingCCAlgo {
  float startTime_;
  float stopTime_;
  float targetTimePrecision_;

  static constexpr int TIME_WHEN_NOT_CONVERGING = 100;
  static constexpr int MAX_NUM_OF_ITERATIONS = 30;
  static constexpr int MIN_NUM_OF_ITERATIONS = 2;
  static constexpr float GLOBAL_TIME_SHIFT = 100;

public:
  EcalUncalibRecHitTimingCCAlgo(const float startTime = -5, const float stopTime = 5, const float targetTimePrecision = 0.001);
  ~EcalUncalibRecHitTimingCCAlgo(){};
  double computeTimeCC(const EcalDataFrame& dataFrame,
                       const std::vector<double>& amplitudes,
                       const EcalPedestals::Item* aped,
                       const EcalMGPAGainRatio* aGain,
                       const FullSampleVector& fullpulse,
                       EcalUncalibratedRecHit& uncalibRecHit,
                       float& errOnTime );

private:
  FullSampleVector interpolatePulse(const FullSampleVector& fullpulse, const float t = 0);
  float computeCC(const std::vector<double>& samples, const FullSampleVector& sigmalTemplate, const float& t);
};

#endif
