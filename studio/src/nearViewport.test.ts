import test from 'node:test';
import assert from 'node:assert/strict';
import { NEAR_MARGIN, whenNearViewport } from './nearViewport.ts';

type Callback = (entries: { isIntersecting: boolean }[]) => void;
/** A stand-in for the browser's observer that the test fires by hand. */
function fakeObserver() {
  const log: string[] = [];
  let fire: Callback = () => {};
  let options: IntersectionObserverInit | undefined;
  class Observer {
    constructor(callback: Callback, init?: IntersectionObserverInit) {
      fire = callback;
      options = init;
    }
    observe(target: Element) {
      log.push(`observe ${(target as unknown as { id: string }).id}`);
    }
    disconnect() {
      log.push('disconnect');
    }
  }
  return {
    Observer: Observer as unknown as typeof IntersectionObserver,
    fire: (visible: boolean) => fire([{ isIntersecting: visible }]),
    log,
    options: () => options,
  };
}
const element = { id: 'thumb' } as unknown as Element;

test('the preview waits until the element nears the visible area, then fires once', () => {
  const fake = fakeObserver();
  let fired = 0;
  whenNearViewport(element, () => fired++, fake.Observer);
  assert.deepEqual(fake.log, ['observe thumb']);
  assert.equal(fake.options()?.rootMargin, NEAR_MARGIN);
  fake.fire(false);
  assert.equal(fired, 0, 'an off-screen element does not fetch');
  fake.fire(true);
  assert.equal(fired, 1);
  assert.deepEqual(fake.log, ['observe thumb', 'disconnect']);
  fake.fire(true);
  assert.equal(fired, 1, 'a late callback fires nothing twice');
});

test('stopping early disconnects the observer and never fires', () => {
  const fake = fakeObserver();
  let fired = 0;
  const stop = whenNearViewport(element, () => fired++, fake.Observer);
  stop();
  assert.deepEqual(fake.log, ['observe thumb', 'disconnect']);
  fake.fire(true);
  assert.equal(fired, 0);
});

test('without an observer the preview downloads at once', () => {
  let fired = 0;
  const stop = whenNearViewport(element, () => fired++, undefined);
  assert.equal(fired, 1);
  stop();
  assert.equal(fired, 1);
});
