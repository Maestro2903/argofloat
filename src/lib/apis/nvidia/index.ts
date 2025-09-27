import { NVIDIA_API_BASE_URL } from '$lib/constants';

export const getNvidiaVersion = async (token: string, urlIdx?: number) => {
	let error = null;

	const res = await fetch(`${NVIDIA_API_BASE_URL}/api/version${urlIdx ? `/${urlIdx}` : ''}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getNvidiaModels = async (token: string = '', urlIdx: null | number = null) => {
	let error = null;

	const res = await fetch(`${NVIDIA_API_BASE_URL}/api/tags${urlIdx !== null ? `/${urlIdx}` : ''}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res?.models ?? [];
};

export const generateNvidiaEmbeddings = async (token: string = '', model: string, text: string) => {
	let error = null;

	const res = await fetch(`${NVIDIA_API_BASE_URL}/api/embeddings`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			model: model,
			input: text
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const generateNvidiaTextCompletion = async (token: string = '', model: string, text: string) => {
	let error = null;

	const res = await fetch(`${NVIDIA_API_BASE_URL}/api/generate`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			model: model,
			prompt: text,
			stream: false
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const generateNvidiaChatCompletion = async (token: string = '', body: object) => {
	const controller = new AbortController();
	let error = null;

	const res = await fetch(`${NVIDIA_API_BASE_URL}/api/chat`, {
		signal: controller.signal,
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(body)
	}).catch((err) => {
		error = err;
		return null;
	});

	if (error) {
		throw error;
	}

	return [res, controller];
};

export const pullNvidiaModel = async (
	token: string,
	tagName: string,
	urlIdx: null | number = null
) => {
	let error = null;
	const controller = new AbortController();

	const res = await fetch(`${NVIDIA_API_BASE_URL}/api/pull${urlIdx !== null ? `/${urlIdx}` : ''}`, {
		signal: controller.signal,
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			name: tagName
		})
	}).catch((err) => {
		error = err;
		return null;
	});

	if (error) {
		throw error;
	}

	return [res, controller];
};

export const deleteNvidiaModel = async (
	token: string,
	tagName: string,
	urlIdx: null | number = null
) => {
	let error = null;

	const res = await fetch(
		`${NVIDIA_API_BASE_URL}/api/delete${urlIdx !== null ? `/${urlIdx}` : ''}`,
		{
			method: 'DELETE',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			},
			body: JSON.stringify({
				name: tagName
			})
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err;
			console.error(err);

			if ('detail' in err) {
				error = err.detail;
			}
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const copyNvidiaModel = async (
	token: string,
	source: string,
	destination: string,
	urlIdx: null | number = null
) => {
	let error = null;

	const res = await fetch(`${NVIDIA_API_BASE_URL}/api/copy${urlIdx !== null ? `/${urlIdx}` : ''}`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({
			source: source,
			destination: destination
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((json) => {
			return json;
		})
		.catch((err) => {
			error = err;
			console.error(err);

			if ('detail' in err) {
				error = err.detail;
			}
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};
