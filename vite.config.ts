import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
	plugins: [
		sveltekit(),
		viteStaticCopy({
			targets: [
				{
					src: 'node_modules/onnxruntime-web/dist/*.jsep.*',

					dest: 'wasm'
				}
			]
		})
	],
	define: {
		APP_VERSION: JSON.stringify(process.env.npm_package_version),
		APP_BUILD_HASH: JSON.stringify(process.env.APP_BUILD_HASH || 'dev-build')
	},
	build: {
		sourcemap: process.env.NODE_ENV === 'development',
		minify: 'terser',
		rollupOptions: {
			output: {
				manualChunks: (id) => {
					// Split vendor libraries into separate chunks
					if (id.includes('node_modules')) {
						if (id.includes('svelte')) return 'svelte';
						if (id.includes('prosemirror') || id.includes('tiptap')) return 'editor';
						if (id.includes('pyodide')) return 'pyodide';
						if (id.includes('transformers') || id.includes('mediapipe')) return 'ml';
						return 'vendor';
					}
				}
			}
		}
	},
	worker: {
		format: 'es'
	},
	esbuild: {
		pure: process.env.ENV === 'dev' ? [] : ['console.log', 'console.debug', 'console.error']
	}
});
