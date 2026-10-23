import { mount } from 'svelte';
import App from './App.svelte';
import { takeLaunchToken } from './launchToken.ts';
import './style.css';
takeLaunchToken(location, history);
mount(App, { target: document.getElementById('app')! });
