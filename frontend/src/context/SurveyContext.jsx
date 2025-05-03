import { createContext, useContext, useState } from 'react';
import axios from 'axios';

const SurveyContext = createContext();

export function useSurvey() {
  return useContext(SurveyContext);
}

export function SurveyProvider({ children }) {
  const [surveyData, setSurveyData] = useState({
    hw_cpu: 1,
    hw_nvidia: 0,
    hw_amd: 0,
    hw_other: 0,
    need_cross_vendor: 0,
    perf_weight: 0.5,
    port_weight: 0.5,
    eco_weight: 0.5,
    pref_directives: 0,
    pref_kernels: 0,
    greenfield: 1,
    gpu_extend: 0,
    cpu_port: 0,
    domain_ai_ml: 0,
    domain_hpc: 0,
    domain_climate: 0,
    domain_embedded: 0,
    domain_graphics: 0,
    domain_data_analytics: 0,
    domain_other: 0,
    lockin_tolerance: 0.5,
    gpu_skill_level: 1
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [step, setStep] = useState(1);

  const updateSurveyData = (newData) => {
    setSurveyData(prevData => ({ ...prevData, ...newData }));
  };

  const submitSurvey = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log("Submitting survey data:", surveyData);
      const response = await axios.post('/api/predict', surveyData);
      console.log("API Response received:", response.data);
      console.log("Framework rankings with probabilities:", response.data.ranking);
      
      // Log the raw values before any processing
      const rankingWithPercentages = response.data.ranking.map(item => ({
        framework: item.framework,
        prob: item.prob,
        percentage: item.prob > 1 ? item.prob : Math.round(item.prob * 100)
      }));
      console.log("Rankings with calculated percentages:", rankingWithPercentages);
      
      setResults(response.data);
      return response.data;
    } catch (err) {
      console.error('Survey submission error:', err);
      setError(err.response?.data?.detail || 'An error occurred while processing your request');
      return null;
    } finally {
      setLoading(false);
    }
  };

  const resetSurvey = () => {
    setSurveyData({
      hw_cpu: 1,
      hw_nvidia: 0,
      hw_amd: 0,
      hw_other: 0,
      need_cross_vendor: 0,
      perf_weight: 0.5,
      port_weight: 0.5,
      eco_weight: 0.5,
      pref_directives: 0,
      pref_kernels: 0,
      greenfield: 1,
      gpu_extend: 0,
      cpu_port: 0,
      domain_ai_ml: 0,
      domain_hpc: 0,
      domain_climate: 0,
      domain_embedded: 0,
      domain_graphics: 0,
      domain_data_analytics: 0,
      domain_other: 0,
      lockin_tolerance: 0.5,
      gpu_skill_level: 1
    });
    setResults(null);
    setStep(1);
  };

  const value = {
    surveyData,
    updateSurveyData,
    results,
    loading,
    error,
    submitSurvey,
    resetSurvey,
    step,
    setStep
  };

  return (
    <SurveyContext.Provider value={value}>
      {children}
    </SurveyContext.Provider>
  );
} 