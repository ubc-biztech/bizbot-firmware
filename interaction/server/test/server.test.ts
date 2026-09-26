import { test } from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { makeServer, type Personality } from '../src/server.js';

const token = 'test-access-token-that-is-at-least-32-characters';
const profile: Personality = {
  id: 'test', name: 'Test', instructions: 'Be concise.', voice: 'marin', greetingsEnabled: true,
  greetingPrompt: 'Say hello', arrivalSeconds: 1, departureSeconds: 10, cooldownSeconds: 30
};

test('session endpoint protects secrets, restricts profiles, and handles upstream errors', async t => {
  let mode = 'success';
  let calls = 0;
  const server = makeServer({
    apiKey: 'permanent-secret-never-returned', accessToken: token, model: 'gpt-realtime', profiles: [profile],
    fetch: (async (url, options) => {
      calls++;
      assert.equal(url, 'https://api.openai.com/v1/realtime/client_secrets');
      assert.equal((options?.headers as Record<string, string>).Authorization, 'Bearer permanent-secret-never-returned');
      const body = JSON.parse(options?.body as string);
      assert.equal(body.session.audio.input.format.rate, 24000);
      assert.equal(body.session.instructions, 'Be concise.');
      assert.equal(body.session.tools.length, 2);
      if (mode === 'failure') return new Response('secret upstream detail', { status: 401 });
      if (mode === 'expired') return Response.json({ value: 'expired', expires_at: 1 });
      if (mode === 'throw') throw new Error('secret details');
      return Response.json({ value: 'ephemeral-only', expires_at: Math.floor(Date.now() / 1000) + 60 });
    }) as typeof fetch
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  const address = server.address();
  assert(address && typeof address !== 'string');
  const url = `http://127.0.0.1:${address.port}/session`;
  const request = (body: string, auth = token) => fetch(url, { method: 'POST', headers: { Authorization: `Bearer ${auth}` }, body });
  assert.equal((await request('{"profileId":"test"}', 'wrong')).status, 401);
  assert.equal(calls, 0);
  assert.equal((await request('{"profileId":"missing"}')).status, 400);
  assert.equal((await request('invalid json')).status, 400);
  assert.equal((await request('x'.repeat(5000))).status, 400);
  const success = await request('{"profileId":"test","model":"untrusted"}');
  assert.equal(success.status, 200);
  assert.equal(success.headers.get('cache-control'), 'no-store');
  const result = await success.json();
  assert.equal(result.clientSecret, 'ephemeral-only');
  assert.equal(result.model, 'gpt-realtime');
  assert(!JSON.stringify(result).includes('permanent-secret'));
  for (mode of ['failure', 'expired', 'throw']) {
    const failure = await request('{"profileId":"test"}');
    assert.equal(failure.status, 502);
    assert(!(await failure.text()).includes('secret'));
  }
  mode = 'success';
  for (let n = 0; n < 6; n++) await request('{"profileId":"test"}');
  assert.equal((await request('{"profileId":"test"}')).status, 429);
});

test('refuses missing server credentials', () => {
  assert.throws(() => makeServer({ apiKey: '', accessToken: token, model: 'gpt-realtime', profiles: [] }));
  assert.throws(() => makeServer({ apiKey: 'key', accessToken: 'short', model: 'gpt-realtime', profiles: [] }));
});
