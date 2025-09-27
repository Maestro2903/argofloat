<script lang="ts">
	// DISABLED: Suggestions functionality has been disabled
	// import Fuse from 'fuse.js';
	// import Bolt from '$lib/components/icons/Bolt.svelte';
	import { onMount, getContext } from 'svelte';
	import { settings, WEBUI_NAME } from '$lib/stores';
	import { WEBUI_VERSION } from '$lib/constants';

	const i18n = getContext('i18n');

	// DISABLED: All suggestion props removed since functionality is disabled
	// export let suggestionPrompts = [];
	// export let className = '';
	// export let inputValue = '';
	// export let onSelect = (e) => {};

	// DISABLED: All suggestion-related logic has been commented out
	// let sortedPrompts = [];
	// const fuseOptions = {
	// 	keys: ['content', 'title'],
	// 	threshold: 0.5
	// };
	// let fuse;
	// let filteredPrompts = [];

	// Force filteredPrompts to always be empty to disable suggestions
	let filteredPrompts = [];

	// DISABLED: All suggestion processing logic
	// $: fuse = new Fuse(sortedPrompts, fuseOptions);
	// $: getFilteredPrompts(inputValue);
	// function arraysEqual(a, b) { ... }
	// const getFilteredPrompts = (inputValue) => { ... }
	// $: if (suggestionPrompts) { ... }
</script>

<!-- DISABLED: Suggestions UI has been disabled -->
<div class="mb-1 flex gap-1 text-xs font-medium items-center text-gray-600 dark:text-gray-400">
	<!-- Always show the app name/version instead of suggestions -->
	<div
		class="flex w-full {$settings?.landingPageMode === 'chat'
			? ' -mt-1'
			: 'text-center items-center justify-center'}  self-start text-gray-600 dark:text-gray-400"
	>
		{$WEBUI_NAME} ‧ v{WEBUI_VERSION}
	</div>
</div>

<!-- DISABLED: Suggestions list completely removed -->
<div class="h-40 w-full">
	<!-- Suggestions are disabled - this area remains empty -->
</div>

<!-- DISABLED: Removed unused CSS for suggestions animations -->
<style>
	/* All suggestion-related styles have been removed */
</style>
