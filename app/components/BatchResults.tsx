'use client'

import { useMemo } from 'react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'
import * as XLSX from 'xlsx'
import styles from './BatchResults.module.css'

interface BatchResult {
  filename: string
  image_type: 'Labeled' | 'Unlabeled'
  true_age?: number | null
  true_gender?: number | null
  predicted_age: number
  predicted_gender: string
  predicted_nationality?: string | null
  nationality_confidence?: number | null
  predicted_emotion?: string | null
  emotion_confidence?: number | null
  age_error?: number | null
  gender_correct?: boolean | null
  gender_confidence: number
}

interface BatchSummary {
  total: number
  labeled_count: number
  unlabeled_count: number
  age_mae?: number | null
  gender_accuracy?: number | null
  avg_confidence?: number | null
  ethnicity_distribution?: Record<string, number>
  emotion_distribution?: Record<string, number>
}

interface BatchResultsProps {
  results: BatchResult[]
  summary: BatchSummary
  errors?: string[]
}

const COLORS = ['#111111', '#2563eb', '#f97316', '#10b981', '#a855f7', '#ef4444', '#14b8a6']

function toCSV(rows: BatchResult[]): string {
  if (!rows.length) return ''
  const header = [
    'filename',
    'image_type',
    'true_age',
    'true_gender',
    'predicted_age',
    'predicted_gender',
    'predicted_nationality',
    'nationality_confidence',
    'predicted_emotion',
    'emotion_confidence',
    'age_error',
    'gender_correct',
    'gender_confidence',
  ]
  const lines = rows.map((r) =>
    [
      r.filename,
      r.image_type,
      r.true_age ?? '',
      r.true_gender ?? '',
      r.predicted_age,
      r.predicted_gender,
      r.predicted_nationality ?? '',
      r.nationality_confidence ?? '',
      r.predicted_emotion ?? '',
      r.emotion_confidence ?? '',
      r.age_error ?? '',
      r.gender_correct === undefined || r.gender_correct === null ? '' : Number(r.gender_correct),
      r.gender_confidence,
    ]
      .map((v) => `${v}`.replace(/"/g, '""'))
      .map((v) => `"${v}"`)
      .join(',')
  )
  return [header.join(','), ...lines].join('\n')
}

function downloadExcel(results: BatchResult[], summary: BatchSummary) {
  const wb = XLSX.utils.book_new()
  const resultSheet = XLSX.utils.json_to_sheet(results)
  XLSX.utils.book_append_sheet(wb, resultSheet, 'Results')

  const summaryRows = [
    ['total', summary.total],
    ['labeled_count', summary.labeled_count],
    ['unlabeled_count', summary.unlabeled_count],
    ['age_mae', summary.age_mae ?? 'N/A'],
    ['gender_accuracy', summary.gender_accuracy ?? 'N/A'],
    ['avg_confidence', summary.avg_confidence ?? 'N/A'],
  ]
  const summarySheet = XLSX.utils.aoa_to_sheet(summaryRows)
  if (summary.ethnicity_distribution && Object.keys(summary.ethnicity_distribution).length) {
    const ethRows = Object.entries(summary.ethnicity_distribution)
    XLSX.utils.sheet_add_aoa(summarySheet, [['ethnicity', 'count'], ...ethRows], { origin: 'D1' })
  }
  if (summary.emotion_distribution && Object.keys(summary.emotion_distribution).length) {
    const emoRows = Object.entries(summary.emotion_distribution)
    XLSX.utils.sheet_add_aoa(summarySheet, [['emotion', 'count'], ...emoRows], { origin: 'G1' })
  }
  XLSX.utils.book_append_sheet(wb, summarySheet, 'Summary')

  const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' })
  const blob = new Blob([wbout], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'batch_results.xlsx'
  a.click()
  URL.revokeObjectURL(url)
}

export default function BatchResults({ results, summary, errors }: BatchResultsProps) {
  const genderData = useMemo(() => {
    const grouped: Record<string, number> = {}
    results.forEach((r) => {
      grouped[r.predicted_gender] = (grouped[r.predicted_gender] || 0) + 1
    })
    return Object.entries(grouped).map(([name, value]) => ({ name, value }))
  }, [results])

  const confidenceData = useMemo(
    () =>
      results.map((r) => ({
        name: r.filename,
        confidence: Math.round((r.gender_confidence || 0) * 1000) / 10,
      })),
    [results]
  )

  const labeledResults = useMemo(() => results.filter((r) => r.image_type === 'Labeled'), [results])

  return (
    <div className={styles.wrapper}>
      <div className={styles.summaryGrid}>
        <div className="metric-card">
          <h4>Total Images</h4>
          <p>{summary.total}</p>
        </div>
        <div className="metric-card">
          <h4>Labeled</h4>
          <p>{summary.labeled_count}</p>
        </div>
        <div className="metric-card">
          <h4>Unlabeled</h4>
          <p>{summary.unlabeled_count}</p>
        </div>
        <div className="metric-card">
          <h4>Age MAE</h4>
          <p>{summary.age_mae !== null && summary.age_mae !== undefined ? summary.age_mae.toFixed(2) : 'N/A'}</p>
        </div>
        <div className="metric-card">
          <h4>Gender Accuracy</h4>
          <p>
            {summary.gender_accuracy !== null && summary.gender_accuracy !== undefined
              ? `${(summary.gender_accuracy * 100).toFixed(1)}%`
              : 'N/A'}
          </p>
        </div>
        <div className="metric-card">
          <h4>Avg Confidence</h4>
          <p>
            {summary.avg_confidence !== null && summary.avg_confidence !== undefined
              ? `${(summary.avg_confidence * 100).toFixed(1)}%`
              : 'N/A'}
          </p>
        </div>
      </div>

      <div className={styles.actions}>
        <button
          className="button"
          onClick={() => {
            const csv = toCSV(results)
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
            const url = URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = url
            link.download = 'batch_results.csv'
            link.click()
            URL.revokeObjectURL(url)
          }}
        >
          Download CSV
        </button>
        <button className="button" onClick={() => downloadExcel(results, summary)}>
          Download Excel
        </button>
      </div>

      <div className={styles.chartsRow}>
        <div className={styles.chartCard}>
          <p className={styles.chartTitle}>Gender distribution</p>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={genderData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={90}>
                {genderData.map((_, idx) => (
                  <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                ))}
              </Pie>
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className={styles.chartCard}>
          <p className={styles.chartTitle}>Confidence per image</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={confidenceData.slice(0, 20)} margin={{ left: -20 }}>
              <XAxis dataKey="name" hide />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
              <Bar dataKey="confidence" fill="#111111" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className={styles.chartNote}>Showing top 20 entries</p>
        </div>
      </div>

      {summary.ethnicity_distribution && Object.keys(summary.ethnicity_distribution).length > 0 && (
        <div className={styles.chartCard}>
          <p className={styles.chartTitle}>Ethnicity distribution</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={Object.entries(summary.ethnicity_distribution).map(([k, v]) => ({ name: k, value: v }))}>
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {summary.emotion_distribution && Object.keys(summary.emotion_distribution).length > 0 && (
        <div className={styles.chartCard}>
          <p className={styles.chartTitle}>Emotion distribution</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={Object.entries(summary.emotion_distribution).map(([k, v]) => ({ name: k, value: v }))}>
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {labeledResults.length > 0 && (
        <div className={styles.tableCard}>
          <div className={styles.tableHeader}>
            <p className={styles.chartTitle}>Labeled vs predicted (sample)</p>
            <p className={styles.chartNote}>First 15 labeled items to sanity check accuracy</p>
          </div>
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>File</th>
                  <th>True Age</th>
                  <th>Pred Age</th>
                  <th>Age Error</th>
                  <th>True Gender</th>
                  <th>Pred Gender</th>
                  <th>Confidence</th>
                  <th>Correct</th>
                </tr>
              </thead>
              <tbody>
                {labeledResults.slice(0, 15).map((r) => (
                  <tr key={r.filename}>
                    <td>{r.filename}</td>
                    <td>{r.true_age ?? '-'}</td>
                    <td>{r.predicted_age}</td>
                    <td>{r.age_error !== null && r.age_error !== undefined ? r.age_error : '-'}</td>
                    <td>
                      {r.true_gender === 0 ? 'Male' : r.true_gender === 1 ? 'Female' : '-'}
                    </td>
                    <td>{r.predicted_gender}</td>
                    <td>{(r.gender_confidence * 100).toFixed(1)}%</td>
                    <td className={r.gender_correct ? styles.good : styles.bad}>
                      {r.gender_correct === null || r.gender_correct === undefined
                        ? '-'
                        : r.gender_correct
                        ? 'Yes'
                        : 'No'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {errors && errors.length > 0 && (
        <div className="warning-box">
          <p>Some files were skipped:</p>
          <ul className={styles.errorList}>
            {errors.map((err) => (
              <li key={err}>{err}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
