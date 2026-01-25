#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lob {

enum class Side { Bid, Ask };

struct Order {
  std::uint64_t id{};
  double price{};
  std::int64_t qty{};
  Side side{};
  std::chrono::high_resolution_clock::time_point timestamp{};
};

struct PriceLevel {
  std::list<Order> orders;
};

struct LevelSnapshot {
  double price{};
  std::int64_t qty{};
};

struct MatchResult {
  std::uint64_t aggressor_id{};
  std::uint64_t resting_id{};
  double price{};
  std::int64_t qty{};
};

class LimitOrderBook {
 public:
  LimitOrderBook() = default;

  void addOrder(std::uint64_t id, double price, std::int64_t qty, Side side);
  bool cancelOrder(std::uint64_t id);
  std::vector<MatchResult> match();
  std::pair<std::vector<LevelSnapshot>, std::vector<LevelSnapshot>> topLevels(
      std::size_t depth = 5) const;

  std::chrono::nanoseconds lastLatency() const { return last_latency_; }

 private:
  using BidMap = std::map<double, PriceLevel, std::greater<double>>;
  using AskMap = std::map<double, PriceLevel, std::less<double>>;

  struct OrderHandle {
    Side side{};
    double price{};
    std::list<Order>::iterator iterator;
  };

  BidMap bids_;
  AskMap asks_;
  std::unordered_map<std::uint64_t, std::unique_ptr<Order>> order_store_;
  std::unordered_map<std::uint64_t, OrderHandle> order_index_;
  std::chrono::nanoseconds last_latency_{0};

  void eraseIfEmpty(Side side, double price);
};

inline void LimitOrderBook::addOrder(std::uint64_t id, double price,
                                     std::int64_t qty, Side side) {
  auto start = std::chrono::high_resolution_clock::now();

  Order order{id, price, qty, side, start};
  if (side == Side::Bid) {
    auto &level = bids_[price];
    level.orders.push_back(order);
    auto iter = std::prev(level.orders.end());
    order_index_[id] = OrderHandle{side, price, iter};
  } else {
    auto &level = asks_[price];
    level.orders.push_back(order);
    auto iter = std::prev(level.orders.end());
    order_index_[id] = OrderHandle{side, price, iter};
  }

  order_store_[id] = std::make_unique<Order>(order);

  last_latency_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
}

inline bool LimitOrderBook::cancelOrder(std::uint64_t id) {
  auto it = order_index_.find(id);
  if (it == order_index_.end()) {
    return false;
  }

  const auto &handle = it->second;
  if (handle.side == Side::Bid) {
    auto level_it = bids_.find(handle.price);
    if (level_it != bids_.end()) {
      level_it->second.orders.erase(handle.iterator);
      eraseIfEmpty(Side::Bid, handle.price);
    }
  } else {
    auto level_it = asks_.find(handle.price);
    if (level_it != asks_.end()) {
      level_it->second.orders.erase(handle.iterator);
      eraseIfEmpty(Side::Ask, handle.price);
    }
  }

  order_index_.erase(it);
  order_store_.erase(id);
  return true;
}

inline std::vector<MatchResult> LimitOrderBook::match() {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<MatchResult> trades;

  while (!bids_.empty() && !asks_.empty()) {
    auto best_bid_it = bids_.begin();
    auto best_ask_it = asks_.begin();

    if (best_bid_it->first < best_ask_it->first) {
      break;
    }

    auto &bid_queue = best_bid_it->second.orders;
    auto &ask_queue = best_ask_it->second.orders;

    if (bid_queue.empty() || ask_queue.empty()) {
      break;
    }

    auto &bid_order = bid_queue.front();
    auto &ask_order = ask_queue.front();

    auto trade_qty = std::min(bid_order.qty, ask_order.qty);
    auto trade_price = best_ask_it->first;

    trades.push_back(
        MatchResult{bid_order.id, ask_order.id, trade_price, trade_qty});

    bid_order.qty -= trade_qty;
    ask_order.qty -= trade_qty;

    if (bid_order.qty == 0) {
      order_index_.erase(bid_order.id);
      order_store_.erase(bid_order.id);
      bid_queue.pop_front();
    }

    if (ask_order.qty == 0) {
      order_index_.erase(ask_order.id);
      order_store_.erase(ask_order.id);
      ask_queue.pop_front();
    }

    eraseIfEmpty(Side::Bid, best_bid_it->first);
    eraseIfEmpty(Side::Ask, best_ask_it->first);
  }

  last_latency_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  return trades;
}

inline std::pair<std::vector<LevelSnapshot>, std::vector<LevelSnapshot>>
LimitOrderBook::topLevels(std::size_t depth) const {
  std::vector<LevelSnapshot> bid_levels;
  std::vector<LevelSnapshot> ask_levels;
  bid_levels.reserve(depth);
  ask_levels.reserve(depth);

  for (auto it = bids_.begin(); it != bids_.end() && bid_levels.size() < depth;
       ++it) {
    std::int64_t total_qty = 0;
    for (const auto &order : it->second.orders) {
      total_qty += order.qty;
    }
    bid_levels.push_back(LevelSnapshot{it->first, total_qty});
  }

  for (auto it = asks_.begin(); it != asks_.end() && ask_levels.size() < depth;
       ++it) {
    std::int64_t total_qty = 0;
    for (const auto &order : it->second.orders) {
      total_qty += order.qty;
    }
    ask_levels.push_back(LevelSnapshot{it->first, total_qty});
  }

  return {bid_levels, ask_levels};
}

inline void LimitOrderBook::eraseIfEmpty(Side side, double price) {
  if (side == Side::Bid) {
    auto it = bids_.find(price);
    if (it != bids_.end() && it->second.orders.empty()) {
      bids_.erase(it);
    }
  } else {
    auto it = asks_.find(price);
    if (it != asks_.end() && it->second.orders.empty()) {
      asks_.erase(it);
    }
  }
}

}  // namespace lob
