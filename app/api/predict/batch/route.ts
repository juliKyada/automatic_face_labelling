import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import { writeFile, unlink } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'

const execAsync = promisify(exec)

async function detectPython(): Promise<string> {
  const candidates = ['python', 'python3', 'py']
  for (const cmd of candidates) {
    try {
      const { stdout } = await execAsync(
        `${cmd} -c "import tensorflow; print('ok')" 2>&1`,
        { timeout: 5000 }
      )
      if (stdout.includes('ok')) {
        return cmd
      }
    } catch (err) {
      continue
    }
  }
  return 'python'
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const items = body?.items

    if (!Array.isArray(items) || items.length === 0) {
      return NextResponse.json(
        { success: false, error: 'No items provided for batch prediction' },
        { status: 400 }
      )
    }

    const tempFile = join(tmpdir(), `batch_${Date.now()}.json`)
    await writeFile(tempFile, JSON.stringify({ items }, null, 2), 'utf-8')

    const projectRoot = process.cwd()
    const pythonScript = join(projectRoot, 'api', 'predict_batch_local.py')
    const pythonCmd = await detectPython()

    const { stdout, stderr } = await execAsync(
      `${pythonCmd} "${pythonScript}" "${tempFile}"`,
      {
        cwd: projectRoot,
        maxBuffer: 20 * 1024 * 1024,
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      }
    )

    await unlink(tempFile).catch(() => {})

    if (!stdout && stderr) {
      throw new Error(stderr)
    }

    const result = JSON.parse(stdout || '{}')
    return NextResponse.json(result)
  } catch (error: any) {
    console.error('Batch predict error:', error)
    return NextResponse.json(
      {
        success: false,
        error: error?.message || 'Batch processing failed. Ensure Python and required packages are installed.'
      },
      { status: 500 }
    )
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    }
  })
}
