'use client'

import { useState, useRef } from 'react'
import axios from 'axios'
import JSZip from 'jszip'
import styles from './BatchUpload.module.css'

export type BatchItem = {
  filename: string
  image: string
  true_age?: number
  true_gender?: number
}

interface BatchUploadProps {
  loading: boolean
  setLoading: (loading: boolean) => void
  onComplete: (payload: { results: any[]; summary: any; errors?: string[] }) => void
}

const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

function parseLabelFromName(name: string): { age?: number; gender?: number } {
  const parts = name.split('_')
  if (parts.length >= 2) {
    const age = Number(parts[0])
    const gender = Number(parts[1])
    if (!Number.isNaN(age) && !Number.isNaN(gender)) {
      return { age, gender }
    }
  }
  return {}
}

function guessMime(ext: string) {
  if (ext.endsWith('png')) return 'image/png'
  if (ext.endsWith('bmp')) return 'image/bmp'
  if (ext.endsWith('tiff') || ext.endsWith('tif')) return 'image/tiff'
  return 'image/jpeg'
}

async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = (err) => reject(err)
    reader.readAsDataURL(file)
  })
}

async function unzipToItems(file: File): Promise<BatchItem[]> {
  const zip = await JSZip().loadAsync(file)
  const items: BatchItem[] = []

  const entries = Object.values(zip.files)
  for (const entry of entries) {
    if (entry.dir) continue
    const lower = entry.name.toLowerCase()
    if (!imageExtensions.some((ext) => lower.endsWith(ext))) continue

    const base64 = await entry.async('base64')
    const ext = lower.split('.').pop() || 'jpg'
    const dataUrl = `data:${guessMime(ext)};base64,${base64}`
    const labels = parseLabelFromName(entry.name)
    items.push({
      filename: entry.name.split('/').pop() || entry.name,
      image: dataUrl,
      true_age: labels.age,
      true_gender: labels.gender,
    })
  }

  return items
}

export default function BatchUpload({ loading, setLoading, onComplete }: BatchUploadProps) {
  const [labeledItems, setLabeledItems] = useState<BatchItem[]>([])
  const [unlabeledItems, setUnlabeledItems] = useState<BatchItem[]>([])
  const [zipProcessing, setZipProcessing] = useState(false)
  const labeledInputRef = useRef<HTMLInputElement>(null)
  const unlabeledInputRef = useRef<HTMLInputElement>(null)
  const zipInputRef = useRef<HTMLInputElement>(null)

  const handleFileList = async (files: FileList | null, target: 'labeled' | 'unlabeled') => {
    if (!files || files.length === 0) return
    const items: BatchItem[] = []

    for (const file of Array.from(files)) {
      const dataUrl = await fileToDataUrl(file)
      const labels = parseLabelFromName(file.name)
      items.push({
        filename: file.name,
        image: dataUrl,
        true_age: target === 'labeled' ? labels.age : undefined,
        true_gender: target === 'labeled' ? labels.gender : undefined,
      })
    }

    if (target === 'labeled') {
      setLabeledItems((prev) => [...prev, ...items])
    } else {
      setUnlabeledItems((prev) => [...prev, ...items])
    }
  }

  const handleZipUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setZipProcessing(true)
    try {
      const allItems: BatchItem[] = []
      for (const file of Array.from(files)) {
        const extracted = await unzipToItems(file)
        allItems.push(...extracted)
      }
      const labeled: BatchItem[] = []
      const unlabeled: BatchItem[] = []
      allItems.forEach((item) => {
        if (item.true_age !== undefined && item.true_gender !== undefined) {
          labeled.push(item)
        } else {
          unlabeled.push(item)
        }
      })
      setLabeledItems((prev) => [...prev, ...labeled])
      setUnlabeledItems((prev) => [...prev, ...unlabeled])
    } catch (err) {
      console.error(err)
      alert('Failed to read zip file. Please verify the archive contents.')
    } finally {
      setZipProcessing(false)
    }
  }

  const handleSubmit = async () => {
    const payload = [...labeledItems, ...unlabeledItems]
    if (payload.length === 0) {
      alert('Please add labeled and/or unlabeled images first.')
      return
    }

    setLoading(true)
    try {
      const { data } = await axios.post('/api/predict/batch', { items: payload })
      onComplete({ results: data.results || [], summary: data.summary || {}, errors: data.errors || [] })
    } catch (err: any) {
      console.error(err)
      alert(err?.response?.data?.error || err?.message || 'Batch prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const clearAll = () => {
    setLabeledItems([])
    setUnlabeledItems([])
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.uploaderRow}>
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <div>
              <p className={styles.cardTitle}>Labeled Images</p>
              <p className={styles.cardSub}>Uses filename pattern age_gender_*.jpg</p>
            </div>
            <span className={styles.count}>{labeledItems.length}</span>
          </div>
          <div className="upload-area" onClick={() => labeledInputRef.current?.click()}>
            <p>Add labeled files</p>
            <p className={styles.help}>JPG, PNG, BMP, TIFF</p>
          </div>
          <input
            type="file"
            accept={imageExtensions.map((ext) => `image/${ext.replace('.', '')}`).join(',')}
            multiple
            ref={labeledInputRef}
            style={{ display: 'none' }}
            onChange={(e) => handleFileList(e.target.files, 'labeled')}
          />
        </div>

        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <div>
              <p className={styles.cardTitle}>Unlabeled Images</p>
              <p className={styles.cardSub}>We will predict labels for these</p>
            </div>
            <span className={styles.count}>{unlabeledItems.length}</span>
          </div>
          <div className="upload-area" onClick={() => unlabeledInputRef.current?.click()}>
            <p>Add unlabeled files</p>
            <p className={styles.help}>JPG, PNG, BMP, TIFF</p>
          </div>
          <input
            type="file"
            accept={imageExtensions.map((ext) => `image/${ext.replace('.', '')}`).join(',')}
            multiple
            ref={unlabeledInputRef}
            style={{ display: 'none' }}
            onChange={(e) => handleFileList(e.target.files, 'unlabeled')}
          />
        </div>

        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <div>
              <p className={styles.cardTitle}>Upload Folder (zip)</p>
              <p className={styles.cardSub}>Drop a zip to auto split labeled/unlabeled</p>
            </div>
            <span className={styles.count}>ZIP</span>
          </div>
          <div className={`upload-area ${styles.zipArea}`} onClick={() => zipInputRef.current?.click()}>
            <p>{zipProcessing ? 'Reading archive...' : 'Drop or click to upload zip'}</p>
            <p className={styles.help}>We scan images recursively</p>
          </div>
          <input
            type="file"
            accept={'.zip'}
            multiple
            ref={zipInputRef}
            style={{ display: 'none' }}
            onChange={(e) => handleZipUpload(e.target.files)}
          />
        </div>
      </div>

      <div className={styles.summaryBar}>
        <div>
          <strong>Total:</strong> {labeledItems.length + unlabeledItems.length} images
        </div>
        <div className={styles.summaryStats}>
          <span>Labeled: {labeledItems.length}</span>
          <span>Unlabeled: {unlabeledItems.length}</span>
        </div>
        <div className={styles.actions}>
          <button className="button" onClick={clearAll} disabled={loading}>
            Clear
          </button>
          <button className="button" onClick={handleSubmit} disabled={loading || zipProcessing}>
            {loading ? 'Processing...' : 'Run Batch Prediction'}
          </button>
        </div>
      </div>
    </div>
  )
}
