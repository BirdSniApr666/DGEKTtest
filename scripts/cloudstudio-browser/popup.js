const statusText = document.querySelector('#status');
const setup = document.querySelector('#setup');
const apps = document.querySelector('#app');
const cancel = document.querySelector('#cancel');

async function send(message) {
  const result = await chrome.runtime.sendMessage(message);
  if (result.error) throw new Error(result.error);
  return result;
}

async function render() {
  const {watch, message, apps: options} = await send({type: 'view'});
  apps.replaceChildren();
  if (options.length > 1) {
    apps.add(new Option('请选择应用', ''));
  }
  for (const [id, title] of options) {
    apps.add(new Option(title ? `${title} (${id})` : id, id));
  }
  statusText.textContent = message || (options.length
    ? '请在训练状态预览页打开此扩展。'
    : '请先打开 Cloud Studio 应用的编辑页。');
  setup.hidden = Boolean(watch) || !options.length;
  cancel.hidden = !watch;
}

async function run(action, button) {
  button.disabled = true;
  try {
    await action();
    await render();
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

setup.addEventListener('submit', event => {
  event.preventDefault();
  run(() => send({
    type: 'start', appId: apps.value,
    stopOnFailure: document.querySelector('#stop-on-failure').checked
  }), event.submitter);
});

cancel.addEventListener('click', () => run(() => send({type: 'cancel'}), cancel));
render().catch(error => { statusText.textContent = error.message; });
