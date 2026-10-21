import { useEffect, useState } from 'react';
import { getBillingStatus, openBillingPortal, startCheckout } from '../api/client';

// Display prices only; Stripe charges whatever the configured Price IDs say.
const PLAN_OPTIONS = [
  { key: 'pro_monthly', label: 'Pro monthly', price: '$12 / month' },
  { key: 'pro_yearly', label: 'Pro yearly', price: '$96 / year' },
  { key: 'thesis_pass', label: 'Thesis Pass', price: '$29 for 30 days' },
];

export default function PlanMenu({ onMessage }) {
  const [status, setStatus] = useState(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      setStatus(await getBillingStatus());
    } catch {
      // backend may not be running yet
    }
  }

  useEffect(() => {
    refresh();
    // Stripe sends the user back with ?billing=success|cancelled.
    const params = new URLSearchParams(window.location.search);
    const result = params.get('billing');
    if (result) {
      onMessage?.(result === 'success' ? '✓ Payment received, thanks!' : 'Checkout cancelled');
      params.delete('billing');
      const query = params.toString();
      window.history.replaceState({}, '', window.location.pathname + (query ? `?${query}` : ''));
    }
    function openMenu() {
      setOpen(true);
      refresh();
    }
    window.addEventListener('billing:refresh', refresh);
    window.addEventListener('billing:open', openMenu);
    return () => {
      window.removeEventListener('billing:refresh', refresh);
      window.removeEventListener('billing:open', openMenu);
    };
  }, []);

  async function redirect(getUrl) {
    setBusy(true);
    try {
      window.location.assign(await getUrl());
    } catch (e) {
      onMessage?.(e.message);
      setBusy(false);
    }
  }

  if (!status) return null;
  const isPro = status.plan === 'pro';
  const offered = PLAN_OPTIONS.filter((option) => status.options.includes(option.key));

  return (
    <div className="user-menu">
      <button
        className={`plan-pill${isPro ? ' plan-pill--pro' : ''}`}
        onClick={() => {
          setOpen((value) => !value);
          refresh();
        }}
        title="Your plan"
      >
        {isPro ? 'Pro' : 'Free'} · {status.ai_requests_used}/{status.ai_requests_limit} AI
      </button>
      {open && (
        <div className="user-menu-dropdown plan-menu">
          <div className="user-menu-name">
            {isPro ? 'DAL Word Pro' : 'Free plan'}
            {status.access_until && isPro && (
              <div className="plan-menu-note">
                Access until {new Date(status.access_until).toLocaleDateString()}
              </div>
            )}
            <div className="plan-menu-note">
              {status.ai_requests_used} of {status.ai_requests_limit} AI requests used this month
            </div>
          </div>
          {!isPro && offered.map((option) => (
            <button
              key={option.key}
              className="plan-menu-option"
              disabled={busy}
              onClick={() => redirect(() => startCheckout(option.key))}
            >
              <span>{option.label}</span>
              <span className="plan-menu-price">{option.price}</span>
            </button>
          ))}
          {!isPro && !offered.length && (
            <div className="plan-menu-note">Upgrades are coming soon.</div>
          )}
          {status.can_manage && (
            <button
              className="plan-menu-option"
              disabled={busy}
              onClick={() => redirect(openBillingPortal)}
            >
              Manage billing
            </button>
          )}
        </div>
      )}
    </div>
  );
}
