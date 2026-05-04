function createBubble(role, text, sources) {
  const bubble = document.createElement('div');
  bubble.classList.add('chat-bubble', role);

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

function formatSource(source) {
  const filename = source.filename || 'Unknown source';
  const chunk = Number.isInteger(source.chunk_index) ? source.chunk_index : '?';
  const details = [`chunk ${chunk}`];
  if (Number.isInteger(source.page_number)) {
    details.push(`page ${source.page_number}`);
  }
  if (source.section_title) {
    details.push(source.section_title);
  }
  return `${filename} (${details.join(', ')})`;
}

function appendSources(bubble, sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return;
  }

  const sourceDetails = document.createElement('details');
  sourceDetails.classList.add('sources');

  const summary = document.createElement('summary');
  summary.textContent = 'Sources';
  sourceDetails.appendChild(summary);

  const list = document.createElement('ul');
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

document.getElementById('chat-form').addEventListener('submit', async function(event) {
  event.preventDefault();

  const queryInput = document.getElementById('query');
  const submitButton = document.getElementById('submit-button');
  const query = queryInput.value.trim();
  if (!query) {
    return;
  }

  submitButton.disabled = true;
  const responseDiv = document.getElementById('response');
  responseDiv.appendChild(createBubble('user', query));
  const aiBubble = createBubble('ai', 'Loading...');
  const aiMessage = aiBubble.querySelector('.message');
  responseDiv.appendChild(aiBubble);
  responseDiv.scrollTop = responseDiv.scrollHeight;
  queryInput.value = '';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      setMessageContent(aiMessage, 'Error: ' + (data.error || 'Request failed.'), 'ai');
    } else {
      setMessageContent(aiMessage, data.response, 'ai');
      appendSources(aiBubble, data.sources);
      updateSessionStatus(data.turn_count);
    }
  } catch (error) {
    setMessageContent(aiMessage, 'Error: ' + error, 'ai');
  } finally {
    submitButton.disabled = false;
    queryInput.focus();
  }
  responseDiv.scrollTop = responseDiv.scrollHeight;
});

document.getElementById('query').addEventListener('keydown', function(event) {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    event.preventDefault();
    document.getElementById('chat-form').requestSubmit();
  }
});

document.getElementById('new-chat-button').addEventListener('click', async function() {
  const responseDiv = document.getElementById('response');
  try {
    await fetch('/api/chat/session', { method: 'DELETE' });
  } catch (error) {
    responseDiv.appendChild(createBubble('ai', 'Error clearing chat: ' + error));
    return;
  }
  responseDiv.textContent = '';
  updateSessionStatus(0);
  document.getElementById('query').focus();
});

function updateSessionStatus(turnCount) {
  const status = document.getElementById('session-status');
  if (!turnCount) {
    status.textContent = 'New chat';
    return;
  }
  status.textContent = `${turnCount} turn${turnCount === 1 ? '' : 's'}`;
}
