"""
Severity Scoring System for Chest X-ray Multi-condition Analysis

Provides severity scoring for multiple pulmonary conditions.
"""

from typing import Dict, List
import numpy as np


class SeverityScorer:
    """Calculate severity scores for chest X-ray conditions."""
    
    def __init__(self):
        """Initialize severity scorer."""
        self.condition_weights = {
            'Pneumonia': 1.0,
            'Pneumothorax': 0.9,
            'Consolidation': 0.8,
            'Edema': 0.85,
            'Mass': 0.7,
            'Nodule': 0.6,
            'Atelectasis': 0.5,
            'Cardiomegaly': 0.75,
            'Infiltration': 0.65,
            'Effusion': 0.7,
            'Emphysema': 0.55,
            'Fibrosis': 0.6,
            'Pleural_Thickening': 0.5,
            'Hernia': 0.3
        }
    
    def calculate_severity_scores(self, predictions: Dict) -> Dict:
        """
        Calculate severity scores for all detected conditions.
        
        Args:
            predictions: Dictionary with condition probabilities
            
        Returns:
            Severity scores and rankings
        """
        condition_scores = {}
        
        for condition, probability in predictions.items():
            if probability > 0.5:  # Threshold for detection
                weight = self.condition_weights.get(condition, 0.5)
                severity_score = probability * weight * 100
                condition_scores[condition] = {
                    'probability': probability,
                    'severity_score': round(severity_score, 2),
                    'severity_level': self._classify_severity(severity_score),
                    'clinical_priority': self._get_priority(severity_score)
                }
        
        # Calculate composite severity index
        if condition_scores:
            composite_score = sum(s['severity_score'] for s in condition_scores.values()) / len(condition_scores)
        else:
            composite_score = 0
        
        # Rank conditions by severity
        ranked_conditions = sorted(
            condition_scores.items(),
            key=lambda x: x[1]['severity_score'],
            reverse=True
        )
        
        return {
            'condition_scores': condition_scores,
            'composite_severity_index': round(composite_score, 2),
            'overall_severity': self._classify_severity(composite_score),
            'ranked_conditions': [c[0] for c in ranked_conditions],
            'total_conditions': len(condition_scores),
            'icu_risk': self._assess_icu_risk(condition_scores, composite_score)
        }
    
    def _classify_severity(self, score: float) -> str:
        """Classify severity level."""
        if score >= 70:
            return "Critical"
        elif score >= 50:
            return "Severe"
        elif score >= 30:
            return "Moderate"
        elif score >= 15:
            return "Mild"
        else:
            return "Minimal"
    
    def _get_priority(self, score: float) -> str:
        """Get clinical priority."""
        if score >= 70:
            return "Immediate"
        elif score >= 50:
            return "Urgent"
        elif score >= 30:
            return "High"
        else:
            return "Routine"
    
    def _assess_icu_risk(self, condition_scores: Dict, composite_score: float) -> Dict:
        """Assess risk of ICU admission."""
        critical_conditions = ['Pneumonia', 'Pneumothorax', 'Edema', 'Consolidation']
        has_critical = any(c in condition_scores for c in critical_conditions)
        
        if composite_score >= 70 or has_critical:
            icu_risk = "High"
            recommendation = "Consider ICU monitoring"
        elif composite_score >= 50:
            icu_risk = "Moderate"
            recommendation = "Monitor closely"
        else:
            icu_risk = "Low"
            recommendation = "Routine care"
        
        return {
            'risk_level': icu_risk,
            'recommendation': recommendation,
            'composite_score': composite_score
        }


class TreatmentRecommendations:
    """Generate treatment recommendations based on conditions."""
    
    def __init__(self):
        """Initialize treatment recommender."""
        self.treatment_guidelines = self._load_guidelines()
    
    def get_recommendations(self, conditions: Dict, severity_scores: Dict) -> Dict:
        """
        Get treatment recommendations for detected conditions.
        
        Args:
            conditions: Detected conditions with probabilities
            severity_scores: Severity scoring results
            
        Returns:
            Treatment recommendations
        """
        recommendations = []
        medications = []
        imaging_followup = []
        specialist_referrals = []
        
        for condition, data in severity_scores.get('condition_scores', {}).items():
            condition_recs = self._get_condition_recommendations(condition, data)
            recommendations.extend(condition_recs.get('recommendations', []))
            medications.extend(condition_recs.get('medications', []))
            imaging_followup.extend(condition_recs.get('imaging', []))
            specialist_referrals.extend(condition_recs.get('referrals', []))
        
        return {
            'general_recommendations': recommendations,
            'medications': list(set(medications)),  # Remove duplicates
            'imaging_followup': imaging_followup,
            'specialist_referrals': list(set(specialist_referrals)),
            'urgency': severity_scores.get('overall_severity', 'Moderate'),
            'next_steps': self._get_next_steps(severity_scores)
        }
    
    def _get_condition_recommendations(self, condition: str, data: Dict) -> Dict:
        """Get recommendations for specific condition."""
        guidelines = {
            'Pneumonia': {
                'recommendations': ['Antibiotic therapy', 'Chest physiotherapy', 'Oxygen support if needed'],
                'medications': ['Antibiotics (based on guidelines)', 'Antipyretics'],
                'imaging': ['Follow-up CXR in 48-72 hours if no improvement'],
                'referrals': ['Pulmonology if severe']
            },
            'Pneumothorax': {
                'recommendations': ['Immediate evaluation', 'Chest tube if large'],
                'medications': [],
                'imaging': ['Immediate repeat CXR', 'CT chest if indicated'],
                'referrals': ['Thoracic surgery']
            },
            'Edema': {
                'recommendations': ['Diuretic therapy', 'Oxygen support', 'Monitor fluid balance'],
                'medications': ['Diuretics', 'ACE inhibitors if cardiac'],
                'imaging': ['Echocardiogram', 'Follow-up CXR'],
                'referrals': ['Cardiology', 'Nephrology if renal']
            },
            'Mass': {
                'recommendations': ['Further imaging', 'Biopsy consideration'],
                'medications': [],
                'imaging': ['CT chest with contrast', 'PET scan if indicated'],
                'referrals': ['Pulmonology', 'Thoracic surgery']
            },
            'Nodule': {
                'recommendations': ['Follow-up imaging per Fleischner guidelines'],
                'medications': [],
                'imaging': ['CT chest', 'Follow-up in 3-6 months'],
                'referrals': ['Pulmonology if suspicious']
            }
        }
        
        return guidelines.get(condition, {
            'recommendations': ['Further evaluation recommended'],
            'medications': [],
            'imaging': ['Follow-up imaging'],
            'referrals': []
        })
    
    def _get_next_steps(self, severity_scores: Dict) -> List[str]:
        """Get next steps based on severity."""
        severity = severity_scores.get('overall_severity', 'Moderate')
        
        if severity == 'Critical':
            return [
                'Immediate clinical evaluation',
                'Consider ICU admission',
                'Initiate treatment protocols',
                'Continuous monitoring'
            ]
        elif severity == 'Severe':
            return [
                'Urgent clinical evaluation',
                'Initiate treatment',
                'Close monitoring',
                'Consider specialist consultation'
            ]
        else:
            return [
                'Clinical evaluation',
                'Initiate appropriate treatment',
                'Follow-up as indicated'
            ]
    
    def _load_guidelines(self) -> Dict:
        """Load treatment guidelines (simplified)."""
        return {}

