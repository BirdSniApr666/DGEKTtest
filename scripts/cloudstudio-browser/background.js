const alarmName = 'cloudstudio-training';
const protocol = 'dgekt-cloudstudio-training-v1';

function appIdFor(tab) {
  if (!tab.url) return null;
  const url = new URL(tab.url);
  if (url.origin !== 'https://cloudstudio.net') return null;
  return url.pathname.match(/^\/a\/([\w-]+)\/(?:edit|learn)(?:\/|$)/)?.[1] ?? null;
}

async function ideTabs() {
  const tabs = await chrome.tabs.query({url: 'https://cloudstudio.net/a/*'});
  return tabs.filter(tab => appIdFor(tab));
}

async function readStatus(tabId) {
  const [{result}] = await chrome.scripting.executeScript({
    target: {tabId},
    func: async () => {
      if (location.protocol !== 'https:') return null;
      const response = await fetch(location.href, {
        cache: 'no-store', redirect: 'error', signal: AbortSignal.timeout(10000)
      });
      return response.ok ? response.json() : null;
    }
  });
  const expectedExit = (result?.state === 'running' && result.exit_code === null)
    || (result?.state === 'succeeded' && result.exit_code === 0)
    || (result?.state === 'failed' && Number.isInteger(result.exit_code) && result.exit_code !== 0);
  if (result?.protocol !== protocol || !/^[a-f0-9]{32}$/.test(result.run_id) || !expectedExit) {
    throw new Error('当前页面没有有效的训练状态。请从端口面板在浏览器中打开状态页。');
  }
  return result;
}

async function finish(message, badge) {
  await chrome.storage.session.set({watch: null, message});
  await chrome.alarms.clear(alarmName);
  await chrome.action.setBadgeText({text: badge});
}

// Keep alarms, popup actions, and tab-close events from changing the watch at the same time.
let pending = Promise.resolve();

function enqueue(operation) {
  const result = pending.then(operation);
  pending = result.catch(() => {});
  return result;
}

async function checkTraining() {
  const {watch} = await chrome.storage.session.get('watch');
  if (!watch) return;

  let status;
  try {
    status = await readStatus(watch.tabId);
  } catch {
    await chrome.action.setBadgeText({text: '?'});
    await chrome.storage.session.set({message: '暂时读不到训练状态，会自动重试。请保持预览页打开。'});
    return;
  }

  if (status.run_id !== watch.runId) {
    await finish('预览页已切换到另一次训练。请为新的训练重新开始监控。', '!');
    return;
  }
  if (status.state === 'running') {
    await chrome.action.setBadgeText({text: 'ON'});
    await chrome.storage.session.set({message: `正在监控 ${watch.appId}，训练仍在运行。`});
    return;
  }

  const outcome = status.exit_code === 0 ? '训练成功结束' : `训练异常退出（退出码 ${status.exit_code}）`;
  if (status.state === 'failed' && !watch.stopOnFailure) {
    await finish(`${outcome}，编辑页已保留。`, '!');
    return;
  }
  const targetTabs = (await ideTabs()).filter(tab => appIdFor(tab) === watch.appId);
  const closed = await Promise.allSettled(targetTabs.map(async ({id}) => {
    const tab = await chrome.tabs.get(id);
    if (appIdFor(tab) === watch.appId) await chrome.tabs.remove(id);
  }));
  if (closed.some(result => result.status === 'rejected')) {
    await chrome.storage.session.set({message: `${outcome}。部分页面暂时未关闭，会自动重试。`});
    return;
  }
  await finish(`${outcome}，该应用在本浏览器的编辑页和学习页已关闭。`, 'OK');
}

async function handleMessage(message) {
  switch (message.type) {
    case 'view': {
      const state = await chrome.storage.session.get(['watch', 'message']);
      const apps = new Map();
      for (const tab of await ideTabs()) apps.set(appIdFor(tab), tab.title);
      return {...state, apps: [...apps.entries()]};
    }
    case 'start': {
      const {watch} = await chrome.storage.session.get('watch');
      if (watch) throw new Error('请先取消当前监控。');
      if (!(await ideTabs()).some(tab => appIdFor(tab) === message.appId)) {
        throw new Error('在本浏览器中找不到所选应用的编辑页或学习页。');
      }
      const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
      if (!tab?.id || appIdFor(tab)) throw new Error('请先切换到训练状态预览页。');
      let status;
      try {
        status = await readStatus(tab.id);
      } catch {
        throw new Error('当前页面没有有效的训练状态。请从端口面板在浏览器中打开状态页。');
      }
      await chrome.storage.session.set({
        watch: {appId: message.appId, tabId: tab.id,
                runId: status.run_id, stopOnFailure: message.stopOnFailure === true},
        message: `正在监控 ${message.appId}。`
      });
      await chrome.alarms.create(alarmName, {periodInMinutes: 0.5});
      await chrome.action.setBadgeText({text: 'ON'});
      await checkTraining();
      return {};
    }
    case 'cancel':
      await finish('已取消监控。', '');
      return {};
    default:
      throw new Error('无法识别此操作。');
  }
}

chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  enqueue(() => handleMessage(message)).then(respond, error => respond({error: error.message}));
  return true;
});

chrome.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === alarmName) enqueue(checkTraining).catch(console.error);
});

chrome.tabs.onRemoved.addListener(tabId => {
  enqueue(async () => {
    const {watch} = await chrome.storage.session.get('watch');
    if (watch?.tabId === tabId) {
      await finish('训练状态页已关闭。请重新打开状态页并开始监控。', '!');
    }
  }).catch(console.error);
});
