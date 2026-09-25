'use strict';
// The app's version, imported by the page AND by the workers.
//
// It exists so the two can check they are the same build. The offline cache
// can end up serving a page from one release and a worker from another, and
// when the messages between them changed (v2.3 did), the worker silently
// ignored every request and Help mode hung on "Finding your best move".
// Now each worker announces its version the moment it starts, and the page
// refuses to rely on one that does not match.
export const APP_VERSION = '2.4.0';
