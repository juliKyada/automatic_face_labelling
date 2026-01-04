'use client'

import { useState } from 'react'
import ImageUpload from './components/ImageUpload'
import ResultsDisplay from './components/ResultsDisplay'
import Navigation from './components/Navigation'
import BatchUpload from './components/BatchUpload'
import BatchResults from './components/BatchResults'
import styles from './page.module.css'

export default function Home() {
  const [activeTab, setActiveTab] = useState('single')
  const [singleResults, setSingleResults] = useState<any>(null)
  const [batchResults, setBatchResults] = useState<any[]>([])
  const [batchSummary, setBatchSummary] = useState<any | null>(null)
  const [batchErrors, setBatchErrors] = useState<string[]>([])
  const [singleLoading, setSingleLoading] = useState(false)
  const [batchLoading, setBatchLoading] = useState(false)

  return (
    <div className={styles.container}>
      <h1 className="main-header">Automatic Facial Image Labelling System</h1>
      
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className={styles.main}>
        {activeTab === 'single' && (
          <div>
            <h2 className="sub-header">Single Image Analysis</h2>
            <ImageUpload 
              onPredict={(result) => setSingleResults(result)} 
              loading={singleLoading}
              setLoading={setSingleLoading}
            />
            {singleResults && <ResultsDisplay results={singleResults} />}
          </div>
        )}
        
        {activeTab === 'batch' && (
          <div>
            <h2 className="sub-header">Batch Processing</h2>
            <BatchUpload
              loading={batchLoading}
              setLoading={setBatchLoading}
              onComplete={({ results, summary, errors }) => {
                setBatchResults(results || [])
                setBatchSummary(summary || null)
                setBatchErrors(errors || [])
              }}
            />
            {batchSummary && (
              <BatchResults
                results={batchResults}
                summary={batchSummary}
                errors={batchErrors}
              />
            )}
          </div>
        )}
        
        {activeTab === 'about' && (
          <div>
            <h2 className="sub-header">About This System</h2>
            <div className="info-box">
              <h3>Key Features</h3>
              <ul style={{ color: '#333', marginLeft: '1.5rem' }}>
                <li style={{ marginBottom: '0.5rem' }}><strong>Multi-label Prediction</strong>: Age, Gender, Ethnicity, and Emotion classification</li>
                <li style={{ marginBottom: '0.5rem' }}><strong>Age Prediction</strong>: Regression-based age estimation (0-100 years)</li>
                <li style={{ marginBottom: '0.5rem' }}><strong>Gender Classification</strong>: Binary classification (Male/Female) with confidence scores</li>
                <li style={{ marginBottom: '0.5rem' }}><strong>Ethnicity Recognition</strong>: Multi-class ethnicity classification (if model available)</li>
                <li style={{ marginBottom: '0.5rem' }}><strong>Emotion Detection</strong>: Facial emotion recognition (if model available)</li>
              </ul>
            </div>
            <div className="info-box">
              <h3>How It Works</h3>
              <ol style={{ color: '#333', marginLeft: '1.5rem' }}>
                <li style={{ marginBottom: '0.5rem' }}>Upload a facial image</li>
                <li style={{ marginBottom: '0.5rem' }}>The system processes the image using deep learning models</li>
                <li style={{ marginBottom: '0.5rem' }}>View comprehensive predictions with confidence scores</li>
                <li style={{ marginBottom: '0.5rem' }}>Export results if needed</li>
              </ol>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

