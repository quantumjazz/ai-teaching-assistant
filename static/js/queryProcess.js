const MODE_CONFIG = {
  ask: {
    label: 'Ask',
    prefix: '',
    placeholder: 'Ask about a specific concept, author, model, or passage...'
  },
  quiz: {
    label: 'Quiz',
    prefix: 'm:',
    placeholder: 'Enter a specific topic for a multiple-choice question...'
  },
  check: {
    label: 'Check answer',
    prefix: 'a:',
    placeholder: 'After a quiz question, paste your answer here...'
  }
};

let activeMode = 'ask';

function createBubble(role, text, sources, options) {
  const bubble = document.createElement('div');
  bubble.classList.add('chat-bubble', role);

  const opts = options || {};
  if (opts.loading) {
    bubble.classList.add('loading');
  }
  if (opts.error) {
    bubble.classList.add('error');
  }

  if (opts.modeLabel) {
    const modeLabel = document.createElement('div');
    modeLabel.classList.add('message-mode');
    modeLabel.textContent = opts.modeLabel;
    bubble.appendChild(modeLabel);
  }

  const message = document.createElement('div');
  message.classList.add('message');
  setMessageContent(message, text, role);

  bubble.appendChild(message);

  appendSources(bubble, sources);

  return bubble;
}

function setMessageContent(message, text, role) {
  message.textContent = '';
  if (role !== 'ai') {
    message.textContent = text;
    return;
  }

  renderMarkdownMessage(message, text);
}

function renderMarkdownMessage(container, text) {
  const lines = String(text || '').split(/\r?\n/);
  let paragraphLines = [];
  let currentList = null;
  let lastListItem = null;

  function flushParagraph() {
    if (paragraphLines.length === 0) {
      return;
    }
    const paragraph = document.createElement('p');
    appendInlineMarkdown(paragraph, paragraphLines.join(' '));
    container.appendChild(paragraph);
    paragraphLines = [];
  }

  function appendListItem(rawLine) {
    const match = rawLine.match(/^(\s*)[-*]\s+(.*)$/);
    const indent = match ? match[1].length : 0;
    const textContent = match ? match[2] : rawLine;

    if (!currentList || indent === 0) {
      currentList = document.createElement('ul');
      container.appendChild(currentList);
      lastListItem = null;
    }

    let targetList = currentList;
    if (indent > 0 && lastListItem) {
      targetList = lastListItem.querySelector(':scope > ul');
      if (!targetList) {
        targetList = document.createElement('ul');
        lastListItem.appendChild(targetList);
      }
    }

    const item = document.createElement('li');
    appendInlineMarkdown(item, textContent);
    targetList.appendChild(item);
    if (indent === 0) {
      lastListItem = item;
    }
  }

  lines.forEach(function(line) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      currentList = null;
      lastListItem = null;
      return;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      flushParagraph();
      appendListItem(line);
      return;
    }

    currentList = null;
    lastListItem = null;
    paragraphLines.push(trimmed);
  });

  flushParagraph();
}

function appendInlineMarkdown(parent, text) {
  const segments = String(text || '').split(/(\*\*[^*]+\*\*)/g);
  segments.forEach(function(segment) {
    if (!segment) {
      return;
    }
    if (segment.startsWith('**') && segment.endsWith('**') && segment.length > 4) {
      const strong = document.createElement('strong');
      strong.textContent = segment.slice(2, -2);
      parent.appendChild(strong);
      return;
    }
    parent.appendChild(document.createTextNode(segment));
  });
}

function sourceLabel(source) {
  return source.document_title || source.filename || 'Unknown source';
}

function formatSource(source) {
  const filename = source.filename || 'Unknown source';
  const details = [];
  if (Number.isInteger(source.page_number)) {
    details.push(`page ${source.page_number}`);
  }
  if (source.section_title) {
    details.push(source.section_title);
  }
  if (Number.isInteger(source.chunk_index)) {
    details.push(`chunk ${source.chunk_index}`);
  }
  return details.length ? `${filename} (${details.join(', ')})` : filename;
}

function appendSources(bubble, sources) {
  const existingSources = bubble.querySelector(':scope > .sources');
  if (existingSources) {
    existingSources.remove();
  }

  if (!Array.isArray(sources) || sources.length === 0) {
    return;
  }

  const sourceDetails = document.createElement('details');
  sourceDetails.classList.add('sources');

  const summary = document.createElement('summary');
  summary.addEventListener('click', function(event) {
    event.preventDefault();
    sourceDetails.open = !sourceDetails.open;
  });
  const count = document.createElement('span');
  count.classList.add('source-count');
  count.textContent = `${sources.length} source${sources.length === 1 ? '' : 's'}`;
  summary.appendChild(count);

  uniqueSourceLabels(sources).slice(0, 3).forEach(function(label) {
    const chip = document.createElement('span');
    chip.classList.add('source-chip');
    chip.textContent = label;
    summary.appendChild(chip);
  });

  sourceDetails.appendChild(summary);

  const list = document.createElement('ul');
  list.classList.add('source-list');
  sources.forEach(function(source) {
    const item = document.createElement('li');
    const title = document.createElement('strong');
    title.textContent = formatSource(source);
    item.appendChild(title);

    if (source.snippet) {
      const snippet = document.createElement('p');
      snippet.textContent = source.snippet;
      item.appendChild(snippet);
    }

    list.appendChild(item);
  });
  sourceDetails.appendChild(list);
  bubble.appendChild(sourceDetails);
}

function uniqueSourceLabels(sources) {
  const labels = [];
  sources.forEach(function(source) {
    const label = sourceLabel(source);
    if (!labels.includes(label)) {
      labels.push(label);
    }
  });
  return labels;
}

function apiQueryFor(displayQuery) {
  const config = MODE_CONFIG[activeMode] || MODE_CONFIG.ask;
  if (!config.prefix) {
    return displayQuery;
  }
  return `${config.prefix} ${displayQuery}`;
}

function displayModeLabel() {
  if (activeMode === 'ask') {
    return '';
  }
  return MODE_CONFIG[activeMode].label;
}

function setMode(mode) {
  if (!MODE_CONFIG[mode]) {
    return;
  }
  activeMode = mode;
  const queryInput = document.getElementById('query');
  queryInput.placeholder = MODE_CONFIG[mode].placeholder;

  document.querySelectorAll('.mode-tab').forEach(function(tab) {
    const isActive = tab.dataset.mode === mode;
    tab.classList.toggle('active', isActive);
    tab.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
}

function updateEmptyState() {
  const emptyState = document.getElementById('empty-state');
  const responseDiv = document.getElementById('response');
  if (!emptyState || !responseDiv) {
    return;
  }
  const hasMessages = responseDiv.querySelector('.chat-bubble') !== null;
  emptyState.classList.toggle('hidden', hasMessages);
}

function scrollConversationToBottom() {
  const responseDiv = document.getElementById('response');
  responseDiv.scrollTop = responseDiv.scrollHeight;
}

function scrollBubbleIntoView(bubble) {
  requestAnimationFrame(function() {
    bubble.scrollIntoView({ block: 'start', behavior: 'smooth' });
  });
}

function setBusy(isBusy) {
  const submitButton = document.getElementById('submit-button');
  const queryInput = document.getElementById('query');
  submitButton.disabled = isBusy;
  queryInput.disabled = isBusy;
  submitButton.textContent = isBusy ? 'Wait' : 'Send';
}

function resizeComposer() {
  const queryInput = document.getElementById('query');
  queryInput.style.height = 'auto';
  queryInput.style.height = `${queryInput.scrollHeight}px`;
}

document.getElementById('chat-form').addEventListener('submit', async function(event) {
  event.preventDefault();

  const queryInput = document.getElementById('query');
  const displayQuery = queryInput.value.trim();
  if (!displayQuery) {
    return;
  }

  setBusy(true);
  const responseDiv = document.getElementById('response');
  responseDiv.appendChild(
    createBubble('user', displayQuery, [], { modeLabel: displayModeLabel() })
  );
  updateEmptyState();

  const aiBubble = createBubble('ai', 'Thinking...', [], { loading: true });
  const aiMessage = aiBubble.querySelector('.message');
  responseDiv.appendChild(aiBubble);
  scrollBubbleIntoView(aiBubble);
  queryInput.value = '';
  resizeComposer();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: apiQueryFor(displayQuery) })
    });
    const data = await res.json();
    aiBubble.classList.remove('loading');
    if (!res.ok || data.error) {
      aiBubble.classList.add('error');
      setMessageContent(aiMessage, 'Error: ' + (data.error || 'Request failed.'), 'ai');
    } else {
      setMessageContent(aiMessage, data.response, 'ai');
      appendSources(aiBubble, data.sources);
      updateSessionStatus(data.turn_count);
    }
  } catch (error) {
    aiBubble.classList.remove('loading');
    aiBubble.classList.add('error');
    setMessageContent(aiMessage, 'Error: ' + error, 'ai');
  } finally {
    setBusy(false);
    queryInput.focus();
  }
  scrollBubbleIntoView(aiBubble);
});

document.getElementById('query').addEventListener('keydown', function(event) {
  if (event.key !== 'Enter') {
    return;
  }
  if (!event.shiftKey || event.metaKey || event.ctrlKey) {
    event.preventDefault();
    document.getElementById('chat-form').requestSubmit();
  }
});

document.getElementById('query').addEventListener('input', resizeComposer);

document.querySelectorAll('.mode-tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    setMode(tab.dataset.mode);
    document.getElementById('query').focus();
  });
});

document.querySelectorAll('.example-action').forEach(function(button) {
  button.addEventListener('click', function() {
    const queryInput = document.getElementById('query');
    setMode(button.dataset.mode || 'ask');
    queryInput.value = button.dataset.prompt || '';
    resizeComposer();
    queryInput.focus();
  });
});

document.querySelectorAll('#new-chat-button').forEach(function(button) {
  button.addEventListener('click', clearChatSession);
});

async function clearChatSession() {
  const responseDiv = document.getElementById('response');
  try {
    await fetch('/api/chat/session', { method: 'DELETE' });
  } catch (error) {
    responseDiv.appendChild(createBubble('ai', 'Error clearing chat: ' + error, [], { error: true }));
    updateEmptyState();
    return;
  }

  responseDiv.querySelectorAll('.chat-bubble').forEach(function(bubble) {
    bubble.remove();
  });
  updateSessionStatus(0);
  updateEmptyState();
  setMode('ask');
  document.getElementById('query').focus();
}

function updateSessionStatus(turnCount) {
  const label = !turnCount
    ? 'New chat'
    : `${turnCount} turn${turnCount === 1 ? '' : 's'}`;
  document.querySelectorAll('#session-status').forEach(function(status) {
    status.textContent = label;
  });
}

setMode('ask');
resizeComposer();
updateEmptyState();
